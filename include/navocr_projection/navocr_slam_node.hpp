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
#include <Eigen/Dense>

#include <memory>
#include <vector>
#include <map>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <deque>
#include <algorithm>

namespace navocr_projection
{

struct Detection
{
  int frame_id;
  cv::Rect bbox;
  double confidence;
  std::string text;  // OCR recognized text
  rclcpp::Time timestamp;
  double depth_m;
  Eigen::Vector3d camera_pos;  // 3D position in camera frame
  Eigen::Vector3d world_pos;   // 3D position in world frame
  bool has_world_pos;
};

// Landmark: Represents a consolidated object in the map
struct Landmark
{
  Eigen::Vector3d mean_position;       // Mean position (μ)
  Eigen::Matrix3d covariance;          // Covariance matrix (Σ)
  int observation_count;               // Number of observations
  std::deque<std::string> text_history; // Recent OCR texts (sliding window)
  std::string representative_text;     // Most common text
  double text_confidence;              // Confidence in text (0-1)
  rclcpp::Time last_updated;           // Last observation time
  int landmark_id;                     // Unique ID
  
  Landmark() 
    : mean_position(Eigen::Vector3d::Zero()),
      covariance(Eigen::Matrix3d::Identity()),
      observation_count(0),
      text_confidence(0.0),
      landmark_id(-1) {}
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
  Eigen::Vector3d projectTo3D(double pixel_u, double pixel_v, double depth_m);
  bool transformToWorld(const Eigen::Vector3d & point_in_camera_frame, const rclcpp::Time & timestamp, 
                        Eigen::Vector3d & point_in_world_frame);
  void publishMarkers();
  
  // Landmark management functions
  void addObservationToLandmarks(const Eigen::Vector3d & world_pos, const std::string & text, 
                                  const rclcpp::Time & timestamp);
  double mahalanobisDistance(const Eigen::Vector3d & point, const Landmark & landmark);
  double levenshteinDistance(const std::string & s1, const std::string & s2);
  double textSimilarity(const std::string & s1, const std::string & s2);
  void updateLandmark(Landmark & landmark, const Eigen::Vector3d & new_pos, const std::string & new_text,
                      const rclcpp::Time & timestamp);
  void createLandmark(const Eigen::Vector3d & pos, const std::string & text, const rclcpp::Time & timestamp);
  void updateRepresentativeText(Landmark & landmark);
  void mergeLandmarks();  // Periodic merging of close landmarks
  
  // Utility functions
  void saveLandmarks();  // Save consolidated landmarks
  
  // Subscribers
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  
  // Publishers
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  
  // TF
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  
  // Parameters
  std::string output_dir_;
  double confidence_threshold_;
  std::string camera_frame_;
  std::string world_frame_;
  
  // Camera info
  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
  double fx_, fy_, cx_, cy_;
  bool camera_info_received_;
  
  // Latest data with timestamp matching
  std::map<rclcpp::Time, cv::Mat> depth_buffer_;
  nav_msgs::msg::Odometry::SharedPtr latest_odom_;
  const size_t buffer_size_ = 50;  // Keep last 50 depths
  
  // Detection storage
  std::vector<Detection> detections_;
  int frame_count_;
  int detection_count_;
  
  // Landmark storage (consolidated objects)
  std::vector<Landmark> landmarks_;
  int next_landmark_id_;
  
  // Landmark parameters (tunable)
  double sensor_noise_std_;         // Sensor noise standard deviation
  int min_observations_;            // Minimum observations for valid landmark
  
  // Landmark parameters (fixed - algorithm constants)
  const double chi2_threshold_;           // χ² threshold for Mahalanobis gate (11.3)
  const double text_similarity_threshold_; // Minimum text similarity (0.6)
  const double acceptance_threshold_;     // Minimum score to accept association (0.5)
  const int text_history_size_;           // Sliding window size for OCR texts (50)
  
  rclcpp::Time last_merge_time_;    // Last time landmarks were merged
  
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
