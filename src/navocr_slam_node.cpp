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
  frame_count_(0),
  detection_count_(0),
  next_landmark_id_(1),
  chi2_threshold_(11.3),              // χ²(3, 0.99) - statistical constant
  text_similarity_threshold_(0.3),    // 30% similarity - relaxed for OCR variability
  acceptance_threshold_(0.5),         // 50% confidence - balanced threshold
  text_history_size_(50)              // Sliding window size
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
}

NavOCRSLAMNode::~NavOCRSLAMNode()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down, saving data...");
  saveLandmarks();
  RCLCPP_INFO(this->get_logger(), "Shutdown complete. Total landmarks: %zu", landmarks_.size());
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
  
  frame_count_++;
  
  RCLCPP_INFO(this->get_logger(), "Frame %d: %zu detections (depth dt=%.1fms)", 
              frame_count_, msg->detections.size(), depth_dt);
  
  for (const auto & det : msg->detections) {
    if (det.results.empty()) continue;
    
    double confidence = det.results[0].hypothesis.score;
    if (confidence < confidence_threshold_) continue;
    
    // Extract OCR label from class_id (PaddleOCR result)
    std::string label = det.results[0].hypothesis.class_id;
    RCLCPP_INFO(this->get_logger(), "Received detection with class_id='%s'", label.c_str());
    if (label.empty()) {
      label = "Unknown";
      RCLCPP_WARN(this->get_logger(), "Empty class_id received, using 'Unknown'");
    }
    
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
    
    // Get depth from synchronized depth image
    double depth_m = 0.0;
    if (center_v >= 0 && center_v < depth_image.rows && 
        center_u >= 0 && center_u < depth_image.cols) {
      uint16_t depth_mm = depth_image.at<uint16_t>(center_v, center_u);
      depth_m = static_cast<double>(depth_mm) / 1000.0;
    }
    
    if (depth_m < 0.1 || depth_m > 10.0) {
      continue;  // Invalid depth
    }
    
    // Project to 3D in camera frame
    Eigen::Vector3d point_in_camera_frame = projectTo3D(center_u, center_v, depth_m);
    
    // Create detection
    Detection detection;
    detection.frame_id = frame_count_;
    detection.bbox = bbox;
    detection.confidence = confidence;
    detection.text = label;  // Store OCR text from vision_msgs
    detection.timestamp = msg->header.stamp;
    detection.depth_m = depth_m;
    detection.camera_pos = point_in_camera_frame;
    detection.has_world_pos = false;
    
    // Transform to world frame
    Eigen::Vector3d point_in_world_frame;
    if (transformToWorld(point_in_camera_frame, msg->header.stamp, point_in_world_frame)) {
      detection.world_pos = point_in_world_frame;
      detection.has_world_pos = true;
      
      // Add to landmark manager (online clustering) with OCR confidence
      addObservationToLandmarks(point_in_world_frame, label, msg->header.stamp, confidence);
      
      RCLCPP_INFO(this->get_logger(),
                  "Detection #%d at world pos (%.2f, %.2f, %.2f), depth=%.2fm, conf=%.2f, text='%s'",
                  detection_count_, point_in_world_frame.x(), point_in_world_frame.y(), point_in_world_frame.z(), 
                  depth_m, confidence, label.c_str());
    } else {
      RCLCPP_INFO(this->get_logger(),
                  "Detection #%d at camera pos (%.2f, %.2f, %.2f), depth=%.2fm, conf=%.2f",
                  detection_count_, point_in_camera_frame.x(), point_in_camera_frame.y(), point_in_camera_frame.z(), 
                  depth_m, confidence);
    }
    
    detections_.push_back(detection);
    detection_count_++;
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
    // Filter by minimum observations
    if (lm.observation_count < min_observations_) {
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
    
    // Size (0.3m cube - larger for better visibility)
    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.3;
    
    // Color based on text confidence (green = high, yellow = medium, red = low)
    if (lm.text_confidence > 0.7) {
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
    } else if (lm.text_confidence > 0.4) {
      marker.color.r = 1.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
    } else {
      marker.color.r = 1.0;
      marker.color.g = 0.5;
      marker.color.b = 0.0;
    }
    marker.color.a = 0.8;
    
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
    text_marker.pose.position.z += 0.3;  // Above the cube
    
    // Display representative text + confidence + observation count
    std::stringstream ss;
    ss << lm.representative_text << " (" 
       << static_cast<int>(lm.text_confidence * 100) << "%, N=" 
       << lm.observation_count << ")";
    text_marker.text = ss.str();
    
    text_marker.scale.z = 0.5;  // Text height (increased for better visibility, especially for Korean)
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;  // White text (better contrast than yellow)
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
    // Create PointStamped in camera frame
    geometry_msgs::msg::PointStamped camera_point_msg;
    camera_point_msg.header.frame_id = camera_frame_;
    camera_point_msg.header.stamp = timestamp;
    camera_point_msg.point.x = point_in_camera_frame.x();
    camera_point_msg.point.y = point_in_camera_frame.y();
    camera_point_msg.point.z = point_in_camera_frame.z();
    
    // Transform to world frame
    geometry_msgs::msg::PointStamped world_point_msg;
    world_point_msg = tf_buffer_->transform(camera_point_msg, world_frame_, 
                                             tf2::durationFromSec(0.5));
    
    point_in_world_frame.x() = world_point_msg.point.x;
    point_in_world_frame.y() = world_point_msg.point.y;
    point_in_world_frame.z() = world_point_msg.point.z;
    
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
                                                double ocr_confidence)
{
  // Step 1: Find candidate landmarks (geometric proximity)
  std::vector<std::pair<int, double>> candidates;  // (index, score)
  
  for (size_t i = 0; i < landmarks_.size(); i++) {
    // Calculate Mahalanobis distance
    double d_squared = mahalanobisDistance(world_pos, landmarks_[i]);
    
    // Chi-squared gate (99% confidence)
    if (d_squared < chi2_threshold_) {
      // Calculate text similarity
      double text_sim = textSimilarity(text, landmarks_[i].representative_text);
      
      // Combined score: 90% geometry, 10% text (relaxed for OCR noise)
      double geo_score = 1.0 - (d_squared / chi2_threshold_);
      double final_score = 0.9 * geo_score + 0.1 * text_sim;
      
      candidates.push_back({i, final_score});
    }
  }
  
  // Step 2: Select best candidate
  if (!candidates.empty()) {
    // Find highest score
    auto best = std::max_element(candidates.begin(), candidates.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; });
    
    int best_idx = best->first;
    double best_score = best->second;
    
    // Step 3: Accept or reject
    if (best_score > acceptance_threshold_) {
      // Update existing landmark
      updateLandmark(landmarks_[best_idx], world_pos, text, timestamp, ocr_confidence);
      RCLCPP_DEBUG(this->get_logger(), 
                   "Observation '%s' (conf=%.2f) merged to Landmark #%d (score=%.2f)", 
                   text.c_str(), ocr_confidence, landmarks_[best_idx].landmark_id, best_score);
      return;
    }
  }
  
  // Step 4: Create new landmark if no match
  createLandmark(world_pos, text, timestamp, ocr_confidence);
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
  
  if (m == 0) return n;
  if (n == 0) return m;
  
  std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
  
  for (size_t i = 0; i <= m; i++) dp[i][0] = i;
  for (size_t j = 0; j <= n; j++) dp[0][j] = j;
  
  for (size_t i = 1; i <= m; i++) {
    for (size_t j = 1; j <= n; j++) {
      int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
      dp[i][j] = std::min({
        dp[i-1][j] + 1,      // deletion
        dp[i][j-1] + 1,      // insertion
        dp[i-1][j-1] + cost  // substitution
      });
    }
  }
  
  return dp[m][n];
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
                                     double ocr_confidence)
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
}

void NavOCRSLAMNode::createLandmark(const Eigen::Vector3d & pos, 
                                     const std::string & text,
                                     const rclcpp::Time & timestamp,
                                     double ocr_confidence)
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
                                     [](const auto& a, const auto& b) { 
                                       return a.second < b.second; 
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
  
  int candidates_checked = 0;
  int candidates_skipped = 0;
  
  // Check all existing landmarks (except itself) for potential merge
  for (size_t i = 0; i < landmarks_.size(); i++) {
    if (i == new_idx) continue;  // Skip self
    
    // Spatial filtering: Skip landmarks too far from robot
    double dist_to_robot = (landmarks_[i].mean_position - robot_pos).norm();
    if (dist_to_robot > merge_search_radius_) {
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
      
      // Merge criteria: text similarity > 0.3 (30%, relaxed for OCR noise)
      if (text_sim > 0.3) {
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

void NavOCRSLAMNode::mergeLandmarks()
{
  // NOTE: This function is no longer called automatically (real-time merging enabled)
  // It can be used for manual cleanup if needed
  
  // Merge landmarks that are too close with similar text
  for (size_t i = 0; i < landmarks_.size(); i++) {
    for (size_t j = i + 1; j < landmarks_.size(); j++) {
      double dist = (landmarks_[i].mean_position - landmarks_[j].mean_position).norm();
      double text_sim = textSimilarity(landmarks_[i].representative_text, 
                                        landmarks_[j].representative_text);
      
      // Merge criteria: close distance (< 1.0m) AND similar text (> 30%)
      if (dist < 1.0 && text_sim > 0.3) {
        // Merge j into i (keep one with more observations)
        if (landmarks_[i].observation_count >= landmarks_[j].observation_count) {
          // Combine observations
          int total_N = landmarks_[i].observation_count + landmarks_[j].observation_count;
          
          // Weighted average of positions
          landmarks_[i].mean_position = 
            (landmarks_[i].mean_position * landmarks_[i].observation_count +
             landmarks_[j].mean_position * landmarks_[j].observation_count) / total_N;
          
          landmarks_[i].observation_count = total_N;
          
          // Merge text histories (with confidence values)
          for (const auto & text_conf_pair : landmarks_[j].text_history) {
            landmarks_[i].text_history.push_back(text_conf_pair);
          }
          while (landmarks_[i].text_history.size() > static_cast<size_t>(text_history_size_)) {
            landmarks_[i].text_history.pop_front();
          }
          
          updateRepresentativeText(landmarks_[i]);
          
          // Remove j
          landmarks_.erase(landmarks_.begin() + j);
          j--;  // Adjust index after erase
          
          RCLCPP_INFO(this->get_logger(), "Merged landmarks (dist=%.2fm, sim=%.2f)", dist, text_sim);
        }
      }
    }
  }
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
  for (const auto & lm : landmarks_) {
    if (lm.observation_count < min_observations_) continue;  // Skip weak landmarks
    
    csv_file << lm.landmark_id << ","
             << lm.mean_position.x() << ","
             << lm.mean_position.y() << ","
             << lm.mean_position.z() << ","
             << lm.representative_text << ","
             << lm.text_confidence << ","
             << lm.observation_count << "\n";
  }
  
  csv_file.close();
  
  RCLCPP_INFO(this->get_logger(), 
              "Saved %zu landmarks to %s", landmarks_.size(), csv_path.c_str());
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
