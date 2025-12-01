#ifndef NAVOCR_PROJECTION__NAVOCR_SLAM_NODE_HPP_
#define NAVOCR_PROJECTION__NAVOCR_SLAM_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <vector>
#include <map>
#include <fstream>
#include <chrono>
#include <filesystem>

namespace navocr_projection
{

struct Detection
{
  int frame_id;
  cv::Rect bbox;
  double confidence;
  rclcpp::Time timestamp;
  double depth_m;
  cv::Point3d camera_pos;  // 3D position in camera frame
  cv::Point3d world_pos;   // 3D position in world frame
  bool has_world_pos;
  cv::Mat image;  // Store image for saving
};

class NavOCRSLAMNode : public rclcpp::Node
{
public:
  explicit NavOCRSLAMNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~NavOCRSLAMNode();

private:
  // Callbacks
  void detectionCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);
  void depthCallback(const sensor_msgs::msg::Image::SharedPtr depth_msg);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr info_msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
  
  // Processing functions
  void processDetections(const vision_msgs::msg::Detection2DArray::SharedPtr detections);
  cv::Point3d projectTo3D(const cv::Point2d & pixel, double depth_m);
  bool transformToWorld(const cv::Point3d & camera_point, const rclcpp::Time & timestamp, 
                        cv::Point3d & world_point);
  void publishMarkers();
  void saveDetectionImage(const cv::Mat & image, const cv::Rect & bbox, int detection_id);
  
  // Utility functions
  void saveDetections();
  
  // Subscribers
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  
  // Publishers
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  
  // TF
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  
  // Parameters
  std::string output_dir_;
  std::string images_dir_;
  double confidence_threshold_;
  bool save_images_;
  std::string camera_frame_;
  std::string world_frame_;
  
  // Camera info
  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
  double fx_, fy_, cx_, cy_;
  bool camera_info_received_;
  
  // Latest data with timestamp matching
  std::map<rclcpp::Time, cv::Mat> image_buffer_;
  std::map<rclcpp::Time, cv::Mat> depth_buffer_;
  nav_msgs::msg::Odometry::SharedPtr latest_odom_;
  const size_t buffer_size_ = 50;  // Keep last 50 images/depths
  
  // Detection storage
  std::vector<Detection> detections_;
  int frame_count_;
  int detection_count_;
  
  // Timestamps
  rclcpp::Time last_process_time_;
  
  // Helper function to find closest timestamp
  template<typename T>
  typename std::map<rclcpp::Time, T>::iterator findClosestTimestamp(
    std::map<rclcpp::Time, T>& buffer, const rclcpp::Time& target_time)
  {
    if (buffer.empty()) {
      return buffer.end();
    }
    
    auto it = buffer.lower_bound(target_time);
    
    if (it == buffer.end()) {
      // target_time is after all timestamps, return last
      return std::prev(buffer.end());
    }
    
    if (it == buffer.begin()) {
      // target_time is before all timestamps, return first
      return it;
    }
    
    // Check which is closer: it or it-1
    auto prev_it = std::prev(it);
    auto dt1 = std::abs((target_time - prev_it->first).nanoseconds());
    auto dt2 = std::abs((it->first - target_time).nanoseconds());
    
    return (dt1 < dt2) ? prev_it : it;
  }
};

}  // namespace navocr_projection

#endif  // NAVOCR_PROJECTION__NAVOCR_SLAM_NODE_HPP_
