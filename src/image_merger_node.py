#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def image_stitching(image1, image2):
    # Görüntüleri gri tonlamaya dönüştür
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # ORB özellik çıkarıcısını oluştur
    orb = cv2.ORB_create()

    # İlk görüntüdeki özellikleri bul ve açıklıkları ve açıları hesapla
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)

    # İkinci görüntüdeki özellikleri bul ve açıklıkları ve açıları hesapla
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # ORB ile özellik eşleştirmesi yap
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # En iyi eşleşmeleri sırala
    matches = sorted(matches, key=lambda x: x.distance)

    # İlk 10 eşleşmeyi seç
    good_matches = matches[:30]

    # Eşleşmelerin görüntü koordinatlarını al
    points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    # Homografi matrisini hesapla
    homography_matrix, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # İkinci görüntüyü birleştir
    merged_image = cv2.warpPerspective(image2, homography_matrix, (image1.shape[1] + image2.shape[1], image2.shape[0]))
    merged_image[0:image1.shape[0], 0:image1.shape[1]] = image1

    return merged_image

class ImageMergerNode(Node):
    def __init__(self):
        super().__init__("image_merger_node")
        self.subscription1 = self.create_subscription(
            Image,
            "camera_1/image_raw",
            self.callback1,
            10
        )
        self.subscription2 = self.create_subscription(
            Image,
            "camera_2/image_raw",
            self.callback2,
            10
        )
        self.cv_bridge = CvBridge()
        self.combined_publisher = self.create_publisher(Image, "concatimage", 10)

        self.image1 = None
        self.image2 = None

    def callback1(self, msg):
        self.image1 = msg
        self.process_images()

    def callback2(self, msg):
        self.image2 = msg
        self.process_images()

    def process_images(self):
        if self.image1 is None or self.image2 is None:
            return

        try:
            cv_image1 = self.cv_bridge.imgmsg_to_cv2(self.image1, "bgr8")
            cv_image2 = self.cv_bridge.imgmsg_to_cv2(self.image2, "bgr8")
        except Exception as e:
            self.get_logger().error("Error converting images: {0}".format(e))
            return

        # İki resmi yatay olarak birleştirin
        #combined_image = np.concatenate((cv_image1, cv_image2), axis=1)
        combined_image = image_stitching(cv_image1, cv_image2)

        # Birleştirilmiş resmi ROS Image mesajına dönüştürün
        combined_msg = self.cv_bridge.cv2_to_imgmsg(combined_image, "bgr8")

        # Birleştirilmiş resmi yayınlayın
        self.combined_publisher.publish(combined_msg)

        # Birleştirilmiş resmi gösterin
        cv2.imshow("Combined Image", combined_image)

        cv2.imshow("Image 1", cv_image1)
        cv2.imshow("Image 2", cv_image2)
        cv2.waitKey(5)

def main(args=None):
    rclpy.init(args=args)
    node = ImageMergerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()