　　　　　　　　　　　　　 Driving behavior database for urban driving situation description
Creator: Takeda Lab, Department of Media Science, Graduate School of Information Science, Nagoya University
■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■□■
───────────────────────────────────────────
■ Media Contents
　・movie/*　　　　　　　      Forward Video DAta
　・records/*　　　　　　　　　Driving Signal Data
　・Readme.txt　　　　　　　 Readme（This File）
───────────────────────────────────────────
■ Driving Signal Data（rosbag file）topic name and description：
- Driver behavior signal
  - /driver/blink, eye_* : Driver's eyes open or not
  - /driver/face_* : Driver's face
  - /driver/gaze_* : Driver's gaze
  - /driver/heart_rate: Driver's heart Rate
- Information obtained from the navigation system
  - /gps_m2/distance/*: Distance to the nearest intersection of your vehicle
  - /gps_m2/road/*: Information on surrounding roads
  - /gps_m2/vehicle/*: Information about ego vehicle such as acceleration
- Information about ego vehicle
  - /vehicle/analog/*: Signal obtained from analog signal of own vehicle
  - /vehicle/extracted_can*: Signal obtained from CAN signal of own vehicle
  - /vehicle/gnss: Location information (latitude/longitude) of own vehicle
Each signal/information is always recorded with a time stamp.
The signal and data types are described in msg_type of each message consisting of topic and data pairs.
■ Other
A ROS environment is required to read this signal data.
ref）http://wiki.ros.org/kinetic/Installation
───────────────────────────────────────────
■ Contact
　〒465-8603
　名古屋市千種区不老町 名古屋大学大学院情報科学研究科 メディア科学専攻 武田研究室 自動車班
　e-mail：nudrive@g.sp.m.is.nagoya-u.ac.jp
　TEL：058-789-3647（武田研秘書席）
End