#include "navocr_projection/navocr_slam_node.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sstream>
#include <iomanip>

namespace navocr_projection
{

NavOCRSLAMNode::NavOCRSLAMNode(const rclcpp::NodeOptions & options)
: Node("navocr_slam_cpp", options),
  frame_count_(0),
  detection_count_(0),
  camera_info_received_(false)
{
  // Declare parameters
  this->declare_parameter("output_dir", "/home/sehyeon/ros2_ws/src/navocr_projection/results_cpp");
  this->declare_parameter("confidence_threshold", 0.3);
  this->declare_parameter("save_images", true);
  this->declare_parameter("camera_frame", "camera_infra1_optical_frame");
  this->declare_parameter("world_frame", "map");
  
  // Get parameters
  output_dir_ = this->get_parameter("output_dir").as_string();
  confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
  save_images_ = this->get_parameter("save_images").as_bool();
  camera_frame_ = this->get_parameter("camera_frame").as_string();
  world_frame_ = this->get_parameter("world_frame").as_string();
  
  // Create output directories
  std::filesystem::create_directories(output_dir_);
  images_dir_ = output_dir_ + "/images";
  if (save_images_) {
    std::filesystem::create_directories(images_dir_);
  }
  
  // Initialize TF
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  
  // Subscribers
  detection_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
    "/navocr/detections", 10,
    std::bind(&NavOCRSLAMNode::detectionCallback, this, std::placeholders::_1));
  
  image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/camera/infra1/image_rect_raw", 10,
    [this](const sensor_msgs::msg::Image::SharedPtr msg) {
      try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        image_buffer_[msg->header.stamp] = cv_ptr->image.clone();
        
        // Keep only last buffer_size_ frames
        while (image_buffer_.size() > buffer_size_) {
          image_buffer_.erase(image_buffer_.begin());
        }
      } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception (image): %s", e.what());
      }
    });
  
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
  
  RCLCPP_INFO(this->get_logger(), "NavOCR-SLAM C++ node started");
  RCLCPP_INFO(this->get_logger(), "Output directory: %s", output_dir_.c_str());
  RCLCPP_INFO(this->get_logger(), "Waiting for NavOCR detections on /navocr/detections");
  RCLCPP_INFO(this->get_logger(), "Publishing markers to /navocr/detection_markers");
  RCLCPP_INFO(this->get_logger(), "Camera frame: %s", camera_frame_.c_str());
  RCLCPP_INFO(this->get_logger(), "World frame: %s", world_frame_.c_str());
}

NavOCRSLAMNode::~NavOCRSLAMNode()
{
  RCLCPP_INFO(this->get_logger(), "Shutting down, saving detections...");
  saveDetections();
  RCLCPP_INFO(this->get_logger(), "Shutdown complete");
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
  
  // Find closest color image by timestamp
  auto image_it = findClosestTimestamp(image_buffer_, msg->header.stamp);
  if (image_it == image_buffer_.end()) {
    RCLCPP_WARN(this->get_logger(), "No color image available");
    return;
  }
  
  cv::Mat& depth_image = depth_it->second;
  cv::Mat& color_image = image_it->second;
  
  // Log timestamp differences for debugging
  rclcpp::Time detection_time(msg->header.stamp);
  auto depth_dt = std::abs((detection_time - depth_it->first).nanoseconds()) / 1e6;  // ms
  auto image_dt = std::abs((detection_time - image_it->first).nanoseconds()) / 1e6;  // ms
  
  frame_count_++;
  
  RCLCPP_INFO(this->get_logger(), "Frame %d: %zu detections (image dt=%.1fms, depth dt=%.1fms)", 
              frame_count_, msg->detections.size(), image_dt, depth_dt);
  
  for (const auto & det : msg->detections) {
    if (det.results.empty()) continue;
    
    double confidence = det.results[0].hypothesis.score;
    if (confidence < confidence_threshold_) continue;
    
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
    cv::Point2d pixel(center_u, center_v);
    cv::Point3d camera_point = projectTo3D(pixel, depth_m);
    
    // Create detection
    Detection detection;
    detection.frame_id = frame_count_;
    detection.bbox = bbox;
    detection.confidence = confidence;
    detection.timestamp = msg->header.stamp;
    detection.depth_m = depth_m;
    detection.camera_pos = camera_point;
    detection.has_world_pos = false;
    
    // Save image with bounding box using synchronized image
    if (save_images_ && !color_image.empty()) {
      cv::Mat img_copy = color_image.clone();
      cv::rectangle(img_copy, bbox, cv::Scalar(0, 255, 0), 2);
      
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << confidence;
      cv::putText(img_copy, ss.str(), cv::Point(bbox.x, bbox.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
      
      detection.image = img_copy;
      saveDetectionImage(img_copy, bbox, detection_count_);
    }
    
    // Transform to world frame
    cv::Point3d world_point;
    if (transformToWorld(camera_point, msg->header.stamp, world_point)) {
      detection.world_pos = world_point;
      detection.has_world_pos = true;
      
      RCLCPP_INFO(this->get_logger(),
                  "Detection #%d at world pos (%.2f, %.2f, %.2f), depth=%.2fm, conf=%.2f",
                  detection_count_, world_point.x, world_point.y, world_point.z, depth_m, confidence);
    } else {
      RCLCPP_INFO(this->get_logger(),
                  "Detection #%d at camera pos (%.2f, %.2f, %.2f), depth=%.2fm, conf=%.2f",
                  detection_count_, camera_point.x, camera_point.y, camera_point.z, depth_m, confidence);
    }
    
    detections_.push_back(detection);
    detection_count_++;
  }
  
  // Publish markers for RViz
  publishMarkers();
}

void NavOCRSLAMNode::processDetections(const vision_msgs::msg::Detection2DArray::SharedPtr detections)
{
  // This function is no longer needed as we process in detectionCallback
}

void NavOCRSLAMNode::publishMarkers()
{
  visualization_msgs::msg::MarkerArray marker_array;
  
  int marker_id = 0;
  for (const auto & det : detections_) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = det.has_world_pos ? world_frame_ : camera_frame_;
    marker.header.stamp = this->now();
    marker.ns = "navocr_detections";
    marker.id = marker_id++;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // Position
    if (det.has_world_pos) {
      marker.pose.position.x = det.world_pos.x;
      marker.pose.position.y = det.world_pos.y;
      marker.pose.position.z = det.world_pos.z;
    } else {
      marker.pose.position.x = det.camera_pos.x;
      marker.pose.position.y = det.camera_pos.y;
      marker.pose.position.z = det.camera_pos.z;
    }
    
    marker.pose.orientation.w = 1.0;
    
    // Size (0.1m cube)
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    
    // Color based on confidence (green to red)
    marker.color.r = 1.0 - det.confidence;
    marker.color.g = det.confidence;
    marker.color.b = 0.0;
    marker.color.a = 0.8;
    
    marker.lifetime = rclcpp::Duration::from_seconds(0);  // Forever
    
    marker_array.markers.push_back(marker);
    
    // Add text marker
    visualization_msgs::msg::Marker text_marker;
    text_marker.header = marker.header;
    text_marker.ns = "navocr_text";
    text_marker.id = marker_id++;
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.pose = marker.pose;
    text_marker.pose.position.z += 0.15;  // Above the cube
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << det.confidence;
    text_marker.text = ss.str();
    
    text_marker.scale.z = 0.05;  // Text height
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;
    
    marker_array.markers.push_back(text_marker);
  }
  
  marker_pub_->publish(marker_array);
  RCLCPP_DEBUG(this->get_logger(), "Published %zu markers", marker_array.markers.size());
}

void NavOCRSLAMNode::saveDetectionImage(const cv::Mat & image, const cv::Rect & bbox, int detection_id)
{
  std::stringstream ss;
  ss << images_dir_ << "/detection_" << std::setw(6) << std::setfill('0') << detection_id << ".jpg";
  cv::imwrite(ss.str(), image);
  RCLCPP_DEBUG(this->get_logger(), "Saved detection image: %s", ss.str().c_str());
}

cv::Point3d NavOCRSLAMNode::projectTo3D(const cv::Point2d & pixel, double depth_m)
{
  cv::Point3d point;
  point.x = (pixel.x - cx_) * depth_m / fx_;
  point.y = (pixel.y - cy_) * depth_m / fy_;
  point.z = depth_m;
  return point;
}

bool NavOCRSLAMNode::transformToWorld(const cv::Point3d & camera_point, 
                                       const rclcpp::Time & timestamp,
                                       cv::Point3d & world_point)
{
  try {
    // Create PointStamped in camera frame
    geometry_msgs::msg::PointStamped camera_point_msg;
    camera_point_msg.header.frame_id = camera_frame_;
    camera_point_msg.header.stamp = timestamp;
    camera_point_msg.point.x = camera_point.x;
    camera_point_msg.point.y = camera_point.y;
    camera_point_msg.point.z = camera_point.z;
    
    // Transform to world frame
    geometry_msgs::msg::PointStamped world_point_msg;
    world_point_msg = tf_buffer_->transform(camera_point_msg, world_frame_, 
                                             tf2::durationFromSec(0.5));
    
    world_point.x = world_point_msg.point.x;
    world_point.y = world_point_msg.point.y;
    world_point.z = world_point_msg.point.z;
    
    return true;
    
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "TF transform failed: %s", ex.what());
    return false;
  }
}

void NavOCRSLAMNode::saveDetections()
{
  if (detections_.empty()) {
    RCLCPP_WARN(this->get_logger(), "No detections to save");
    return;
  }
  
  // Generate timestamp
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream timestamp_ss;
  timestamp_ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
  
  std::string csv_path = output_dir_ + "/detections_" + timestamp_ss.str() + ".csv";
  std::ofstream csv_file(csv_path);
  
  if (!csv_file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", csv_path.c_str());
    return;
  }
  
  // Write header
  csv_file << "frame,bbox_x,bbox_y,bbox_w,bbox_h,confidence,"
           << "timestamp_sec,timestamp_nsec,depth_m,"
           << "camera_x,camera_y,camera_z,"
           << "world_x,world_y,world_z,has_world_pos\n";
  
  // Write data
  for (const auto & det : detections_) {
    csv_file << det.frame_id << ","
             << det.bbox.x << "," << det.bbox.y << "," 
             << det.bbox.width << "," << det.bbox.height << ","
             << det.confidence << ","
             << det.timestamp.seconds() << ","
             << det.timestamp.nanoseconds() << ","
             << det.depth_m << ","
             << det.camera_pos.x << "," << det.camera_pos.y << "," << det.camera_pos.z << ",";
    
    if (det.has_world_pos) {
      csv_file << det.world_pos.x << "," << det.world_pos.y << "," << det.world_pos.z << ",1\n";
    } else {
      csv_file << "0,0,0,0\n";
    }
  }
  
  csv_file.close();
  
  RCLCPP_INFO(this->get_logger(), 
              "Saved %zu detections to %s", detections_.size(), csv_path.c_str());
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
