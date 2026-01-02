#include "navocr_projection/navocr_slam_node.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sstream>
#include <iomanip>

namespace navocr_projection
{

NavOCRSLAMNode::NavOCRSLAMNode(const rclcpp::NodeOptions & options)
: Node("navocr_slam_cpp", options),
  camera_info_received_(false),
  next_landmark_id_(1),
  // Statistical thresholds - optimized for 0.5x playback speed
  // For 1.0x: chi2=11.3, text_sim=0.4, acceptance=0.5
  chi2_threshold_(13.5),              // χ²(3, 0.99) relaxed - allow more spatial variance
  text_similarity_threshold_(0.35),   // 35% similarity - more lenient for duplicate detection
  acceptance_threshold_(0.55),        // 55% confidence - slightly stricter for merging
  text_history_size_(50),             // Sliding window size
  min_confidence_for_output_(0.25),   // 25% minimum confidence for display & save
  marker_cube_size_(0.3),             // Marker cube size in meters
  marker_transparency_(0.8),          // Marker alpha value (0.0-1.0)
  text_marker_height_(0.5),           // Text marker height in meters
  confidence_threshold_high_(0.7),    // High confidence threshold (green)
  confidence_threshold_medium_(0.4)   // Medium confidence threshold (yellow)
{
  // Declare essential parameters only
  this->declare_parameter("output_dir", "/home/sehyeon/ros2_ws/src/navocr_projection/results_cpp");
  this->declare_parameter("confidence_threshold", 0.3);
  this->declare_parameter("camera_frame", "camera_infra1_optical_frame");
  this->declare_parameter("world_frame", "map");
  
  // Tunable landmark parameters
  this->declare_parameter("sensor_noise_std", 0.3);      // 30cm depth noise (sensor-specific)
  this->declare_parameter("min_observations", 3);        // Minimum observations for valid landmark
  this->declare_parameter("merge_search_radius", 10.0);  // Search radius for spatial filtering (meters)
  
  // Get parameters
  output_dir_ = this->get_parameter("output_dir").as_string();
  confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
  camera_frame_ = this->get_parameter("camera_frame").as_string();
  world_frame_ = this->get_parameter("world_frame").as_string();
  
  sensor_noise_std_ = this->get_parameter("sensor_noise_std").as_double();
  min_observations_ = this->get_parameter("min_observations").as_int();
  merge_search_radius_ = this->get_parameter("merge_search_radius").as_double();
  
  // Removed: last_merge_time_ initialization (immediate merging enabled)
  confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
  camera_frame_ = this->get_parameter("camera_frame").as_string();
  world_frame_ = this->get_parameter("world_frame").as_string();
  
  // Create output directory
  std::filesystem::create_directories(output_dir_);
  
  // Initialize TF
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  
  // Subscribers
  detection_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
    "/navocr/detections", 10,
    std::bind(&NavOCRSLAMNode::detectionCallback, this, std::placeholders::_1));
  
  depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/depth/image_rect_raw", 10,
    std::bind(&NavOCRSLAMNode::depthCallback, this, std::placeholders::_1));
  
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera/infra1/camera_info", 10,
    std::bind(&NavOCRSLAMNode::cameraInfoCallback, this, std::placeholders::_1));
  
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/odom", 10,
    std::bind(&NavOCRSLAMNode::odomCallback, this, std::placeholders::_1));
  
  // Publishers
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/navocr/detection_markers", 10);
  
  RCLCPP_INFO(this->get_logger(), "NavOCR-SLAM node started");
  RCLCPP_INFO(this->get_logger(), "Output directory: %s", output_dir_.c_str());
  RCLCPP_INFO(this->get_logger(), "Waiting for NavOCR detections on /navocr/detections");
  RCLCPP_INFO(this->get_logger(), "Publishing markers to /navocr/detection_markers");
  RCLCPP_INFO(this->get_logger(), "Camera frame: %s", camera_frame_.c_str());
  RCLCPP_INFO(this->get_logger(), "World frame: %s", world_frame_.c_str());
  RCLCPP_INFO(this->get_logger(), "=== Landmark Manager Settings ===");
  RCLCPP_INFO(this->get_logger(), "  Sensor noise: %.2fm", sensor_noise_std_);
  RCLCPP_INFO(this->get_logger(), "  Min observations: %d", min_observations_);
  RCLCPP_INFO(this->get_logger(), "  Merge search radius: %.1fm", merge_search_radius_);
  RCLCPP_INFO(this->get_logger(), "  Chi² threshold: %.1f (fixed)", chi2_threshold_);
  RCLCPP_INFO(this->get_logger(), "  Text similarity threshold: %.2f (fixed)", text_similarity_threshold_);
  RCLCPP_INFO(this->get_logger(), "  Acceptance threshold: %.2f (fixed)", acceptance_threshold_);
  RCLCPP_INFO(this->get_logger(), "  Min confidence for output: %.2f (RViz & CSV)", min_confidence_for_output_);
}

NavOCRSLAMNode::~NavOCRSLAMNode()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down, saving data...");
  saveLandmarks();
  saveReprojectionSummary();
  RCLCPP_INFO(this->get_logger(), "Shutdown complete. Total landmarks: %zu", landmarks_.size());
}

// Helper function to get color based on confidence level
std_msgs::msg::ColorRGBA NavOCRSLAMNode::getConfidenceColor(double confidence) const
{
  std_msgs::msg::ColorRGBA color;
  color.a = 1.0;  // Full opacity
  
  if (confidence >= confidence_threshold_high_) {
    // High confidence: Green
    color.r = 0.0;
    color.g = 1.0;
    color.b = 0.0;
  } else if (confidence >= confidence_threshold_medium_) {
    // Medium confidence: Yellow
    color.r = 1.0;
    color.g = 1.0;
    color.b = 0.0;
  } else {
    // Low confidence: Orange
    color.r = 1.0;
    color.g = 0.5;
    color.b = 0.0;
  }
  
  return color;
}

void NavOCRSLAMNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  if (!camera_info_received_) {
    camera_info_ = msg;
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];
    camera_info_received_ = true;
    
    RCLCPP_INFO(this->get_logger(), 
                "Camera info received - fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
                fx_, fy_, cx_, cy_);
  }
}

void NavOCRSLAMNode::depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
    depth_buffer_[msg->header.stamp] = cv_ptr->image.clone();
    
    // Keep only last buffer_size_ frames
    while (depth_buffer_.size() > buffer_size_) {
      depth_buffer_.erase(depth_buffer_.begin());
    }
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception (depth): %s", e.what());
  }
}

void NavOCRSLAMNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  latest_odom_ = msg;
}

void NavOCRSLAMNode::detectionCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
  if (!camera_info_received_) {
    return;
  }
  
  // Find closest depth image by timestamp
  auto depth_it = findClosestTimestamp(depth_buffer_, msg->header.stamp);
  if (depth_it == depth_buffer_.end()) {
    RCLCPP_WARN(this->get_logger(), "No depth image available");
    return;
  }
  
  cv::Mat& depth_image = depth_it->second;
  
  // Log timestamp difference for debugging
  rclcpp::Time detection_time(msg->header.stamp);
  auto depth_dt = std::abs((detection_time - depth_it->first).nanoseconds()) / 1e6;  // ms
  
  RCLCPP_INFO(this->get_logger(), "Processing %zu detections (depth dt=%.1fms)", 
              msg->detections.size(), depth_dt);
  
  for (const auto & det : msg->detections) {
    if (det.results.empty()) continue;
    
    double confidence = det.results[0].hypothesis.score;
    if (confidence < confidence_threshold_) continue;
    
    // Extract OCR label from class_id (PaddleOCR result)
    std::string label = det.results[0].hypothesis.class_id;
    
    // Text validation: filter out invalid OCR results
    if (label.empty()) {
      RCLCPP_DEBUG(this->get_logger(), "Skipping empty text detection");
      continue;
    }
    
    // Remove leading/trailing whitespace
    label.erase(0, label.find_first_not_of(" \t\n\r"));
    label.erase(label.find_last_not_of(" \t\n\r") + 1);
    
    if (label.empty()) {
      RCLCPP_DEBUG(this->get_logger(), "Skipping whitespace-only text");
      continue;
    }
    
    // Filter out too short text (likely noise)
    if (label.length() < 2) {
      RCLCPP_DEBUG(this->get_logger(), "Skipping too short text: '%s'", label.c_str());
      continue;
    }
    
    // Filter out too long text (likely OCR error)
    if (label.length() > 50) {
      RCLCPP_DEBUG(this->get_logger(), "Skipping too long text (len=%zu)", label.length());
      continue;
    }
    
    RCLCPP_INFO(this->get_logger(), "Valid detection with text='%s' (conf=%.2f)", label.c_str(), confidence);
    
    // Get bounding box center
    int center_u = static_cast<int>(det.bbox.center.position.x);
    int center_v = static_cast<int>(det.bbox.center.position.y);
    int size_x = static_cast<int>(det.bbox.size_x);
    int size_y = static_cast<int>(det.bbox.size_y);
    
    cv::Rect bbox(
      center_u - size_x/2, 
      center_v - size_y/2,
      size_x,
      size_y
    );
    
    // Get depth from synchronized depth image (average around center for stability)
    double depth_m = 0.0;
    
    // Define sampling window size (10x10 pixels around center)
    const int window_size = 10;
    const int half_window = window_size / 2;
    
    // Calculate sampling region boundaries (ensure within image bounds)
    int u_start = std::max(0, center_u - half_window);
    int u_end = std::min(depth_image.cols, center_u + half_window);
    int v_start = std::max(0, center_v - half_window);
    int v_end = std::min(depth_image.rows, center_v + half_window);
    
    // Extract ROI using OpenCV (vectorized operation - much faster)
    cv::Rect roi(u_start, v_start, u_end - u_start, v_end - v_start);
    cv::Mat depth_roi = depth_image(roi);
    
    // Create mask for valid depth values (100mm ~ 10000mm)
    cv::Mat valid_mask = (depth_roi > 100) & (depth_roi < 10000);
    
    // Compute mean of valid pixels using OpenCV mean() - hardware optimized
    cv::Scalar mean_depth = cv::mean(depth_roi, valid_mask);
    int valid_count = cv::countNonZero(valid_mask);
    
    // Use average if enough valid pixels, otherwise skip
    if (valid_count < (window_size * window_size / 4)) {
      RCLCPP_DEBUG(this->get_logger(), 
                   "Insufficient valid depth pixels (%d/%d) at (%d, %d)", 
                   valid_count, window_size * window_size, center_u, center_v);
      continue;  // Not enough valid depth data
    }
    
    depth_m = mean_depth[0] / 1000.0;  // Convert mm to m
    
    if (depth_m < 0.1 || depth_m > 10.0) {
      continue;  // Invalid depth
    }
    
    // Project to 3D in camera frame
    Eigen::Vector3d point_in_camera_frame = projectTo3D(center_u, center_v, depth_m);
    
    // Transform to world frame and add to landmark manager
    Eigen::Vector3d point_in_world_frame;
    if (transformToWorld(point_in_camera_frame, msg->header.stamp, point_in_world_frame)) {
      // Add to landmark manager (online clustering) with OCR confidence and bbox info
      addObservationToLandmarks(point_in_world_frame, label, msg->header.stamp, confidence,
                                center_u, center_v, size_x, size_y);

      RCLCPP_INFO(this->get_logger(),
                  "Detection at world pos (%.2f, %.2f, %.2f), depth=%.2fm, conf=%.2f, text='%s'",
                  point_in_world_frame.x(), point_in_world_frame.y(), point_in_world_frame.z(),
                  depth_m, confidence, label.c_str());
    } else {
      RCLCPP_DEBUG(this->get_logger(),
                   "Failed to transform to world frame: camera pos (%.2f, %.2f, %.2f), depth=%.2fm",
                   point_in_camera_frame.x(), point_in_camera_frame.y(), point_in_camera_frame.z(), 
                   depth_m);
    }
  }
  
  // Removed: Periodic merging (5-second timer)
  // Merging now happens immediately in checkAndMergeNewLandmark()
  
  // Publish markers for RViz (now shows consolidated landmarks)
  publishMarkers();
}

void NavOCRSLAMNode::publishMarkers()
{
  visualization_msgs::msg::MarkerArray marker_array;
  
  int marker_id = 0;
  
  // Publish consolidated landmarks (not individual detections)
  for (const auto & lm : landmarks_) {
    // Filter 1: Minimum observations
    if (lm.observation_count < min_observations_) {
      continue;
    }
    
    // Filter 2: Minimum confidence (unified threshold)
    if (lm.text_confidence < min_confidence_for_output_) {
      continue;
    }
    
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = world_frame_;
    marker.header.stamp = this->now();
    marker.ns = "navocr_landmarks";
    marker.id = marker_id++;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Position (mean of observations)
    marker.pose.position.x = lm.mean_position.x();
    marker.pose.position.y = lm.mean_position.y();
    marker.pose.position.z = lm.mean_position.z();
    marker.pose.orientation.w = 1.0;
    
    // Size from member variable
    marker.scale.x = marker_cube_size_;
    marker.scale.y = marker_cube_size_;
    marker.scale.z = marker_cube_size_;
    
    // Set color based on confidence level using helper function
    marker.color = getConfidenceColor(lm.text_confidence);
    marker.color.a = marker_transparency_;
    
    marker.lifetime = rclcpp::Duration::from_seconds(0);  // Forever
    
    marker_array.markers.push_back(marker);
    
    // Add text marker with representative text + confidence
    visualization_msgs::msg::Marker text_marker;
    text_marker.header = marker.header;
    text_marker.ns = "navocr_landmark_text";
    text_marker.id = marker_id++;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.pose = marker.pose;
    text_marker.pose.position.z += marker_cube_size_;  // Above the cube
    
    // Display representative text + confidence + observation count
    std::stringstream ss;
    ss << lm.representative_text << " (" 
       << static_cast<int>(lm.text_confidence * 100) << "%, N=" 
       << lm.observation_count << ")";
    text_marker.text = ss.str();
    
    text_marker.scale.z = text_marker_height_;
    
    // White text for better contrast
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;
    
    marker_array.markers.push_back(text_marker);
  }
  
  marker_pub_->publish(marker_array);
  RCLCPP_INFO(this->get_logger(), "Published %zu landmark markers (total landmarks: %zu)", 
              marker_array.markers.size() / 2, landmarks_.size());
}

Eigen::Vector3d NavOCRSLAMNode::projectTo3D(double pixel_u, double pixel_v, double depth_m)
{
  Eigen::Vector3d point;
  point.x() = (pixel_u - cx_) * depth_m / fx_;
  point.y() = (pixel_v - cy_) * depth_m / fy_;
  point.z() = depth_m;
  return point;
}

bool NavOCRSLAMNode::transformToWorld(const Eigen::Vector3d & point_in_camera_frame, 
                                       const rclcpp::Time & timestamp,
                                       Eigen::Vector3d & point_in_world_frame)
{
  try {
    // Create PointStamped message for detected text location in camera frame
    geometry_msgs::msg::PointStamped detected_text_point_in_camera_frame_msg;
    detected_text_point_in_camera_frame_msg.header.frame_id = camera_frame_;
    detected_text_point_in_camera_frame_msg.header.stamp = timestamp;
    detected_text_point_in_camera_frame_msg.point.x = point_in_camera_frame.x();
    detected_text_point_in_camera_frame_msg.point.y = point_in_camera_frame.y();
    detected_text_point_in_camera_frame_msg.point.z = point_in_camera_frame.z();
    
    // Transform detected text location to world frame
    geometry_msgs::msg::PointStamped detected_text_point_in_world_frame_msg;
    detected_text_point_in_world_frame_msg = tf_buffer_->transform(
      detected_text_point_in_camera_frame_msg, world_frame_, tf2::durationFromSec(0.5));
    
    point_in_world_frame.x() = detected_text_point_in_world_frame_msg.point.x;
    point_in_world_frame.y() = detected_text_point_in_world_frame_msg.point.y;
    point_in_world_frame.z() = detected_text_point_in_world_frame_msg.point.z;
    
    return true;
    
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "TF transform failed: %s", ex.what());
    return false;
  }
}

// ========================================================================
// Landmark Management Functions
// ========================================================================

void NavOCRSLAMNode::addObservationToLandmarks(const Eigen::Vector3d & world_pos,
                                                const std::string & text,
                                                const rclcpp::Time & timestamp,
                                                double ocr_confidence,
                                                double bbox_center_u, double bbox_center_v,
                                                int bbox_size_x, int bbox_size_y)
{
  // Step 1: Find candidate landmarks (geometric proximity)
  // Store: (index, combined_score, text_similarity) to avoid recalculation
  std::vector<std::tuple<int, double, double>> candidates;  // (index, score, text_sim)

  for (size_t i = 0; i < landmarks_.size(); i++) {
    // Calculate Mahalanobis distance
    double d_squared = mahalanobisDistance(world_pos, landmarks_[i]);

    // Chi-squared gate (99% confidence)
    if (d_squared < chi2_threshold_) {
      // Calculate text similarity ONCE and cache it
      double text_sim = textSimilarity(text, landmarks_[i].representative_text);

      // Combined score: 90% geometry, 10% text (relaxed for OCR noise)
      double geo_score = 1.0 - (d_squared / chi2_threshold_);
      double final_score = 0.9 * geo_score + 0.1 * text_sim;

      candidates.push_back({i, final_score, text_sim});  // Store text_sim for reuse
    }
  }

  // Step 2: Select best candidate
  if (!candidates.empty()) {
    // Find highest score
    auto best = std::max_element(candidates.begin(), candidates.end(),
                                   [](const auto& candidate_a, const auto& candidate_b) {
                                     return std::get<1>(candidate_a) < std::get<1>(candidate_b);
                                   });

    int best_idx = std::get<0>(*best);
    double best_score = std::get<1>(*best);
    double cached_text_sim = std::get<2>(*best);  // Reuse cached similarity

    // Step 3: Accept or reject
    if (best_score > acceptance_threshold_) {
      // Update existing landmark
      updateLandmark(landmarks_[best_idx], world_pos, text, timestamp, ocr_confidence,
                     bbox_center_u, bbox_center_v, bbox_size_x, bbox_size_y);
      RCLCPP_DEBUG(this->get_logger(),
                   "Observation '%s' (conf=%.2f) merged to Landmark #%d (score=%.2f, text_sim=%.2f)",
                   text.c_str(), ocr_confidence, landmarks_[best_idx].landmark_id, best_score, cached_text_sim);
      return;
    }
  }

  // Step 4: Create new landmark if no match
  createLandmark(world_pos, text, timestamp, ocr_confidence,
                 bbox_center_u, bbox_center_v, bbox_size_x, bbox_size_y);
  RCLCPP_INFO(this->get_logger(),
              "New landmark #%d created for '%s' (conf=%.2f) at (%.2f, %.2f, %.2f)",
              next_landmark_id_ - 1, text.c_str(), ocr_confidence, world_pos.x(), world_pos.y(), world_pos.z());
}

double NavOCRSLAMNode::mahalanobisDistance(const Eigen::Vector3d & point, const Landmark & landmark)
{
  // Difference vector
  Eigen::Vector3d diff = point - landmark.mean_position;
  
  // d² = diffᵀ × Σ⁻¹ × diff
  Eigen::Matrix3d cov_inv = landmark.covariance.inverse();
  
  double d_squared = diff.transpose() * cov_inv * diff;
  
  return d_squared;
}

double NavOCRSLAMNode::levenshteinDistance(const std::string & s1, const std::string & s2)
{
  const size_t m = s1.size();
  const size_t n = s2.size();
  
  // Base cases
  if (m == 0) return n;
  if (n == 0) return m;
  
  // Early termination: If length difference is too large, similarity will be low
  // Skip expensive computation if difference > 60% of max length
  size_t max_len = std::max(m, n);
  if (std::abs(static_cast<int>(m) - static_cast<int>(n)) > static_cast<int>(max_len * 0.6)) {
    return std::abs(static_cast<int>(m) - static_cast<int>(n));
  }
  
  // Space optimization: Use 1D rolling arrays instead of 2D (75% memory reduction)
  // We only need the previous row to compute the current row
  std::vector<int> prev_row(n + 1);
  std::vector<int> curr_row(n + 1);
  
  // Initialize first row
  for (size_t j = 0; j <= n; j++) {
    prev_row[j] = j;
  }
  
  // Compute edit distance using rolling arrays
  for (size_t i = 1; i <= m; i++) {
    curr_row[0] = i;  // First column
    
    for (size_t j = 1; j <= n; j++) {
      int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
      curr_row[j] = std::min({
        prev_row[j] + 1,      // deletion
        curr_row[j-1] + 1,    // insertion
        prev_row[j-1] + cost  // substitution
      });
    }
    
    // Swap rows for next iteration (O(1) operation with pointers)
    std::swap(prev_row, curr_row);
  }
  
  return prev_row[n];  // Result is in prev_row after final swap
}

double NavOCRSLAMNode::textSimilarity(const std::string & s1, const std::string & s2)
{
  if (s1.empty() || s2.empty()) return 0.0;
  
  double dist = levenshteinDistance(s1, s2);
  double max_len = std::max(s1.size(), s2.size());
  
  return 1.0 - (dist / max_len);
}

void NavOCRSLAMNode::updateLandmark(Landmark & landmark,
                                     const Eigen::Vector3d & new_pos,
                                     const std::string & new_text,
                                     const rclcpp::Time & timestamp,
                                     double ocr_confidence,
                                     double bbox_center_u, double bbox_center_v,
                                     int bbox_size_x, int bbox_size_y)
{
  // Welford's online algorithm for mean update
  landmark.observation_count++;
  int N = landmark.observation_count;

  Eigen::Vector3d delta = new_pos - landmark.mean_position;

  // Update mean: μ_new = μ_old + δ/N
  landmark.mean_position += delta / N;

  // Simple covariance update (could use Welford's for variance too)
  // For now, keep initial covariance or use simple moving average

  // Add text to history (sliding window) with OCR confidence
  landmark.text_history.push_back({new_text, ocr_confidence});
  if (landmark.text_history.size() > static_cast<size_t>(text_history_size_)) {
    landmark.text_history.pop_front();
  }

  // Update representative text (weighted by confidence)
  updateRepresentativeText(landmark);

  landmark.last_updated = timestamp;

  // Store observation for reprojection error calculation
  Observation obs;
  obs.timestamp = timestamp;

  // Get camera pose from TF
  try {
    auto transform = tf_buffer_->lookupTransform(world_frame_, camera_frame_, timestamp, tf2::durationFromSec(0.1));
    obs.camera_position = Eigen::Vector3d(
      transform.transform.translation.x,
      transform.transform.translation.y,
      transform.transform.translation.z
    );
    obs.camera_orientation = Eigen::Quaterniond(
      transform.transform.rotation.w,
      transform.transform.rotation.x,
      transform.transform.rotation.y,
      transform.transform.rotation.z
    );
  } catch (tf2::TransformException & ex) {
    // Fallback: use latest odom if TF fails
    if (latest_odom_) {
      obs.camera_position = Eigen::Vector3d(
        latest_odom_->pose.pose.position.x,
        latest_odom_->pose.pose.position.y,
        latest_odom_->pose.pose.position.z
      );
      obs.camera_orientation = Eigen::Quaterniond(
        latest_odom_->pose.pose.orientation.w,
        latest_odom_->pose.pose.orientation.x,
        latest_odom_->pose.pose.orientation.y,
        latest_odom_->pose.pose.orientation.z
      );
    } else {
      // Skip observation if no pose available
      return;
    }
  }

  obs.bbox_center_u = bbox_center_u;
  obs.bbox_center_v = bbox_center_v;
  obs.bbox_size_x = bbox_size_x;
  obs.bbox_size_y = bbox_size_y;

  // Calculate reprojection error immediately (online calculation)
  // Transform landmark position to camera frame
  Eigen::Matrix3d R_world_to_camera = obs.camera_orientation.toRotationMatrix().transpose();
  Eigen::Vector3d t_world_to_camera = -R_world_to_camera * obs.camera_position;
  Eigen::Vector3d P_camera = R_world_to_camera * landmark.mean_position + t_world_to_camera;

  // Calculate reprojection error if landmark is in front of camera
  if (P_camera.z() > 0) {
    // Project to image plane
    double u_proj = fx_ * P_camera.x() / P_camera.z() + cx_;
    double v_proj = fy_ * P_camera.y() / P_camera.z() + cy_;

    // Compute reprojection error
    obs.reprojection_error = std::sqrt((u_proj - bbox_center_u) * (u_proj - bbox_center_u) +
                                       (v_proj - bbox_center_v) * (v_proj - bbox_center_v));
  } else {
    // Landmark behind camera - set invalid error
    obs.reprojection_error = -1.0;
  }

  landmark.observations.push_back(obs);
}

void NavOCRSLAMNode::createLandmark(const Eigen::Vector3d & pos,
                                     const std::string & text,
                                     const rclcpp::Time & timestamp,
                                     double ocr_confidence,
                                     double bbox_center_u, double bbox_center_v,
                                     int bbox_size_x, int bbox_size_y)
{
  Landmark lm;
  lm.mean_position = pos;
  lm.observation_count = 1;
  lm.landmark_id = next_landmark_id_++;
  lm.last_updated = timestamp;

  // Initial covariance based on sensor noise
  double var = sensor_noise_std_ * sensor_noise_std_;
  lm.covariance = Eigen::Matrix3d::Identity();
  lm.covariance(0, 0) = var;
  lm.covariance(1, 1) = var;
  lm.covariance(2, 2) = var * 9.0;  // z-axis (depth) has 3x more noise

  lm.text_history.push_back({text, ocr_confidence});
  lm.representative_text = text;
  lm.text_confidence = ocr_confidence;  // Initial confidence from OCR

  // Store initial observation for reprojection error calculation
  Observation obs;
  obs.timestamp = timestamp;

  // Get camera pose from TF
  try {
    auto transform = tf_buffer_->lookupTransform(world_frame_, camera_frame_, timestamp, tf2::durationFromSec(0.1));
    obs.camera_position = Eigen::Vector3d(
      transform.transform.translation.x,
      transform.transform.translation.y,
      transform.transform.translation.z
    );
    obs.camera_orientation = Eigen::Quaterniond(
      transform.transform.rotation.w,
      transform.transform.rotation.x,
      transform.transform.rotation.y,
      transform.transform.rotation.z
    );
  } catch (tf2::TransformException & ex) {
    // Fallback: use latest odom if TF fails
    if (latest_odom_) {
      obs.camera_position = Eigen::Vector3d(
        latest_odom_->pose.pose.position.x,
        latest_odom_->pose.pose.position.y,
        latest_odom_->pose.pose.position.z
      );
      obs.camera_orientation = Eigen::Quaterniond(
        latest_odom_->pose.pose.orientation.w,
        latest_odom_->pose.pose.orientation.x,
        latest_odom_->pose.pose.orientation.y,
        latest_odom_->pose.pose.orientation.z
      );
    } else {
      // Cannot get camera pose - create landmark without observation
      RCLCPP_WARN(this->get_logger(), "Cannot get camera pose for initial observation of landmark #%d", lm.landmark_id);
      landmarks_.push_back(lm);
      checkAndMergeNewLandmark(landmarks_.size() - 1);
      return;
    }
  }

  obs.bbox_center_u = bbox_center_u;
  obs.bbox_center_v = bbox_center_v;
  obs.bbox_size_x = bbox_size_x;
  obs.bbox_size_y = bbox_size_y;

  // Calculate reprojection error immediately (online calculation)
  Eigen::Matrix3d R_world_to_camera = obs.camera_orientation.toRotationMatrix().transpose();
  Eigen::Vector3d t_world_to_camera = -R_world_to_camera * obs.camera_position;
  Eigen::Vector3d P_camera = R_world_to_camera * lm.mean_position + t_world_to_camera;

  if (P_camera.z() > 0) {
    double u_proj = fx_ * P_camera.x() / P_camera.z() + cx_;
    double v_proj = fy_ * P_camera.y() / P_camera.z() + cy_;
    obs.reprojection_error = std::sqrt((u_proj - bbox_center_u) * (u_proj - bbox_center_u) +
                                       (v_proj - bbox_center_v) * (v_proj - bbox_center_v));
  } else {
    obs.reprojection_error = -1.0;
  }

  lm.observations.push_back(obs);

  landmarks_.push_back(lm);

  // Immediately check if this new landmark should merge with existing ones
  checkAndMergeNewLandmark(landmarks_.size() - 1);
}

void NavOCRSLAMNode::updateRepresentativeText(Landmark & landmark)
{
  if (landmark.text_history.empty()) {
    landmark.text_confidence = 0.0;
    return;
  }
  
  // Weighted voting: Sum confidence scores for each unique text
  std::map<std::string, double> weighted_scores;  // text -> sum of confidences
  
  for (const auto & [text, conf] : landmark.text_history) {
    weighted_scores[text] += conf;  // Accumulate confidence
  }
  
  // Find text with highest weighted score
  auto max_elem = std::max_element(weighted_scores.begin(), weighted_scores.end(),
                                     [](const auto& text_score_pair_a, const auto& text_score_pair_b) { 
                                       return text_score_pair_a.second < text_score_pair_b.second; 
                                     });
  
  landmark.representative_text = max_elem->first;
  
  // Confidence = (sum of matching confidences) / (total confidence sum)
  double total_confidence = 0.0;
  for (const auto & [text, score] : weighted_scores) {
    total_confidence += score;
  }
  
  landmark.text_confidence = max_elem->second / total_confidence;
  
  RCLCPP_DEBUG(this->get_logger(), 
               "Landmark #%d: representative='%s', confidence=%.2f (weighted from %zu observations)",
               landmark.landmark_id, landmark.representative_text.c_str(), 
               landmark.text_confidence, landmark.text_history.size());
}

Eigen::Vector3d NavOCRSLAMNode::getCurrentRobotPosition()
{
  // Try to get robot position from latest odometry
  if (latest_odom_) {
    return Eigen::Vector3d(
      latest_odom_->pose.pose.position.x,
      latest_odom_->pose.pose.position.y,
      latest_odom_->pose.pose.position.z
    );
  }
  
  // Fallback: try to get transform from world frame to camera frame
  try {
    auto transform = tf_buffer_->lookupTransform(world_frame_, camera_frame_, tf2::TimePointZero);
    return Eigen::Vector3d(
      transform.transform.translation.x,
      transform.transform.translation.y,
      transform.transform.translation.z
    );
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Could not get robot position: %s", ex.what());
  }
  
  // Last resort: return origin
  return Eigen::Vector3d::Zero();
}

void NavOCRSLAMNode::checkAndMergeNewLandmark(size_t new_idx)
{
  if (new_idx >= landmarks_.size()) {
    RCLCPP_WARN(this->get_logger(), "Invalid landmark index %zu", new_idx);
    return;
  }
  
  Landmark& new_lm = landmarks_[new_idx];
  
  // Get current robot position for spatial filtering
  Eigen::Vector3d robot_pos = getCurrentRobotPosition();
  
  // Calculate robot distance to new landmark ONCE (outside loop)
  double robot_dist_new = (new_lm.mean_position - robot_pos).norm();
  
  int candidates_checked = 0;
  int candidates_skipped = 0;
  
  // Check all existing landmarks (except itself) for potential merge
  for (size_t i = 0; i < landmarks_.size(); i++) {
    if (i == new_idx) continue;  // Skip self
    
    // Spatial filtering: Use pre-calculated robot_dist_new
    double robot_dist_existing = (landmarks_[i].mean_position - robot_pos).norm();
    
    // Skip if distance difference exceeds search radius
    if (std::abs(robot_dist_new - robot_dist_existing) > merge_search_radius_) {
      candidates_skipped++;
      continue;  // Skip distant landmarks
    }
    
    candidates_checked++;
    
    // Calculate Euclidean distance
    double dist = (new_lm.mean_position - landmarks_[i].mean_position).norm();
    
    // Merge criteria: distance < 1.0m (relaxed from 0.5m)
    if (dist < 1.0) {
      // Calculate text similarity
      double text_sim = textSimilarity(new_lm.representative_text, 
                                        landmarks_[i].representative_text);
      
      // Merge criteria: text similarity > 0.4 (40%, balanced for OCR noise)
      if (text_sim > 0.4) {
        // Merge new landmark into existing landmark i
        int total_N = landmarks_[i].observation_count + new_lm.observation_count;
        
        // Weighted average of positions
        landmarks_[i].mean_position = 
          (landmarks_[i].mean_position * landmarks_[i].observation_count +
           new_lm.mean_position * new_lm.observation_count) / total_N;
        
        landmarks_[i].observation_count = total_N;
        
        // Merge text histories (with confidence values)
        for (const auto & text_conf_pair : new_lm.text_history) {
          landmarks_[i].text_history.push_back(text_conf_pair);
        }
        while (landmarks_[i].text_history.size() > static_cast<size_t>(text_history_size_)) {
          landmarks_[i].text_history.pop_front();
        }
        
        updateRepresentativeText(landmarks_[i]);
        
        RCLCPP_INFO(this->get_logger(), 
                    "New landmark #%d immediately merged with #%d (dist=%.2fm, text_sim=%.2f, checked=%d, skipped=%d)",
                    new_lm.landmark_id, landmarks_[i].landmark_id, dist, text_sim, 
                    candidates_checked, candidates_skipped);
        
        // Remove the new landmark (it's been merged)
        landmarks_.erase(landmarks_.begin() + new_idx);
        
        return;  // Merge complete, exit
      }
    }
  }
  
  // No merge candidate found - landmark remains as is
  RCLCPP_DEBUG(this->get_logger(), 
               "New landmark #%d created (no merge candidate found, checked=%d, skipped=%d)",
               new_lm.landmark_id, candidates_checked, candidates_skipped);
}

void NavOCRSLAMNode::saveLandmarks()
{
  if (landmarks_.empty()) {
    RCLCPP_WARN(this->get_logger(), "No landmarks to save");
    return;
  }
  
  // Generate timestamp
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream timestamp_ss;
  timestamp_ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
  
  std::string csv_path = output_dir_ + "/landmarks_" + timestamp_ss.str() + ".csv";
  std::ofstream csv_file(csv_path);
  
  if (!csv_file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open landmarks CSV: %s", csv_path.c_str());
    return;
  }
  
  // Write header
  csv_file << "landmark_id,x,y,z,representative_text,text_confidence,observation_count\n";
  
  csv_file << std::fixed << std::setprecision(6);
  
  // Write data
  int saved_count = 0;
  for (const auto & lm : landmarks_) {
    // Filter 1: Minimum observations
    if (lm.observation_count < min_observations_) continue;
    
    // Filter 2: Minimum confidence (unified threshold)
    if (lm.text_confidence < min_confidence_for_output_) continue;
    
    csv_file << lm.landmark_id << ","
             << lm.mean_position.x() << ","
             << lm.mean_position.y() << ","
             << lm.mean_position.z() << ","
             << lm.representative_text << ","
             << lm.text_confidence << ","
             << lm.observation_count << "\n";
    saved_count++;
  }
  
  csv_file.close();
  
  RCLCPP_INFO(this->get_logger(),
              "Saved %d/%zu landmarks to %s (filtered by N>=%d and conf>=%.2f)",
              saved_count, landmarks_.size(), csv_path.c_str(),
              min_observations_, min_confidence_for_output_);
}

// ========================================================================
// Reprojection Error Calculation Functions
// ========================================================================

void NavOCRSLAMNode::computeReprojectionErrors()
{
  std::string error_file = output_dir_ + "/NavOCR_reprojection_error.txt";
  std::ofstream outfile(error_file);

  if (!outfile.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open %s", error_file.c_str());
    return;
  }

  outfile << "# NavOCR Reprojection Error Data" << std::endl;
  outfile << "# Format: landmark_id,text,timestamp,error_px" << std::endl;

  int total_observations = 0;

  for (const auto& lm : landmarks_) {
    // Filter: only valid landmarks
    if (lm.observation_count < min_observations_) continue;
    if (lm.text_confidence < min_confidence_for_output_) continue;

    // For each observation, use pre-calculated reprojection error
    for (const auto& obs : lm.observations) {
      // Skip invalid observations (landmark was behind camera)
      if (obs.reprojection_error < 0) continue;

      // Write pre-calculated error to file
      outfile << lm.landmark_id << ",\"" << lm.representative_text << "\","
              << obs.timestamp.nanoseconds() / 1e9 << "," << obs.reprojection_error << std::endl;

      total_observations++;
    }
  }

  outfile.close();

  RCLCPP_INFO(this->get_logger(),
              "Wrote %d pre-calculated reprojection errors across %zu landmarks",
              total_observations, landmarks_.size());
  RCLCPP_INFO(this->get_logger(), "Saved to %s", error_file.c_str());
}

void NavOCRSLAMNode::saveReprojectionSummary()
{
  // First, compute reprojection errors
  computeReprojectionErrors();

  // Read and parse the error file
  std::string error_file = output_dir_ + "/NavOCR_reprojection_error.txt";
  std::ifstream infile(error_file);

  if (!infile.is_open()) {
    RCLCPP_WARN(this->get_logger(), "Could not open %s for summary generation", error_file.c_str());
    return;
  }

  std::map<int, std::vector<double>> landmark_errors;  // landmark_id -> errors
  std::map<int, std::string> landmark_texts;
  std::string line;

  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') continue;

    // Parse CSV: landmark_id,text,timestamp,error
    std::istringstream iss(line);
    std::string id_str, text, timestamp_str, error_str;

    std::getline(iss, id_str, ',');
    std::getline(iss, text, ',');
    std::getline(iss, timestamp_str, ',');
    std::getline(iss, error_str, ',');

    // Remove quotes from text
    if (text.front() == '"') text = text.substr(1, text.length() - 2);

    int landmark_id = std::stoi(id_str);
    double error = std::stod(error_str);

    landmark_errors[landmark_id].push_back(error);
    landmark_texts[landmark_id] = text;
  }
  infile.close();

  // Compute statistics
  std::string summary_file = output_dir_ + "/NavOCR_reprojection_summary.txt";
  std::ofstream summary(summary_file);

  summary << "=== NavOCR Reprojection Error Summary ===" << std::endl << std::endl;
  summary << "landmark_id,text,num_observations,mean_error,std_error,min_error,max_error" << std::endl;

  std::vector<double> all_errors;
  std::vector<double> weighted_errors;
  std::vector<double> weights;

  for (const auto& pair : landmark_errors) {
    int landmark_id = pair.first;
    const std::vector<double>& errors = pair.second;

    if (errors.empty()) continue;

    // Statistics
    double mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();

    double std_error = 0.0;
    for (double e : errors) std_error += (e - mean_error) * (e - mean_error);
    std_error = std::sqrt(std_error / errors.size());

    double min_error = *std::min_element(errors.begin(), errors.end());
    double max_error = *std::max_element(errors.begin(), errors.end());

    summary << landmark_id << ",\"" << landmark_texts[landmark_id] << "\","
            << errors.size() << "," << mean_error << "," << std_error << ","
            << min_error << "," << max_error << std::endl;

    // Accumulate
    for (double e : errors) all_errors.push_back(e);

    double weight = std::sqrt(static_cast<double>(errors.size()));
    weighted_errors.push_back(mean_error);
    weights.push_back(weight);
  }

  // Global statistics
  summary << std::endl << "=== Global Statistics ===" << std::endl;

  double global_mean = std::accumulate(all_errors.begin(), all_errors.end(), 0.0) / all_errors.size();

  double weighted_mean = 0.0, total_weight = 0.0;
  for (size_t i = 0; i < weighted_errors.size(); i++) {
    weighted_mean += weighted_errors[i] * weights[i];
    total_weight += weights[i];
  }
  weighted_mean /= total_weight;

  // Histogram
  std::vector<int> histogram(8, 0);
  for (double e : all_errors) {
    if (e < 1) histogram[0]++;
    else if (e < 2) histogram[1]++;
    else if (e < 3) histogram[2]++;
    else if (e < 4) histogram[3]++;
    else if (e < 5) histogram[4]++;
    else if (e < 7) histogram[5]++;
    else if (e < 10) histogram[6]++;
    else histogram[7]++;
  }

  summary << "Total observations: " << all_errors.size() << std::endl;
  summary << "Total landmarks: " << landmark_errors.size() << std::endl;
  summary << "Overall mean error: " << global_mean << " pixels" << std::endl;
  summary << "Weighted mean error: " << weighted_mean << " pixels (weighted by sqrt(num_obs))" << std::endl;

  summary << std::endl << "=== Error Distribution ===" << std::endl;
  summary << "0-1 px:   " << histogram[0] << " (" << (100.0*histogram[0]/all_errors.size()) << "%)" << std::endl;
  summary << "1-2 px:   " << histogram[1] << " (" << (100.0*histogram[1]/all_errors.size()) << "%)" << std::endl;
  summary << "2-3 px:   " << histogram[2] << " (" << (100.0*histogram[2]/all_errors.size()) << "%)" << std::endl;
  summary << "3-4 px:   " << histogram[3] << " (" << (100.0*histogram[3]/all_errors.size()) << "%)" << std::endl;
  summary << "4-5 px:   " << histogram[4] << " (" << (100.0*histogram[4]/all_errors.size()) << "%)" << std::endl;
  summary << "5-7 px:   " << histogram[5] << " (" << (100.0*histogram[5]/all_errors.size()) << "%)" << std::endl;
  summary << "7-10 px:  " << histogram[6] << " (" << (100.0*histogram[6]/all_errors.size()) << "%)" << std::endl;
  summary << ">10 px:   " << histogram[7] << " (" << (100.0*histogram[7]/all_errors.size()) << "%)" << std::endl;

  int cumulative_3px = histogram[0] + histogram[1] + histogram[2];
  int cumulative_5px = cumulative_3px + histogram[3] + histogram[4];
  summary << std::endl << "Cumulative (<=3px): " << (100.0*cumulative_3px/all_errors.size()) << "%" << std::endl;
  summary << "Cumulative (<=5px): " << (100.0*cumulative_5px/all_errors.size()) << "%" << std::endl;

  summary.close();

  RCLCPP_INFO(this->get_logger(), "=== NavOCR Reprojection Error Summary ===");
  RCLCPP_INFO(this->get_logger(), "Total observations: %zu", all_errors.size());
  RCLCPP_INFO(this->get_logger(), "Total landmarks: %zu", landmark_errors.size());
  RCLCPP_INFO(this->get_logger(), "Weighted mean error: %.2f pixels", weighted_mean);
  RCLCPP_INFO(this->get_logger(), "Summary saved to %s", summary_file.c_str());
}

}  // namespace navocr_projection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<navocr_projection::NavOCRSLAMNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
