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
    good_matches = matches[:10]

    # Eşleşmelerin görüntü koordinatlarını al
    points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    # Homografi matrisini hesapla
    homography_matrix, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # İkinci görüntüyü birleştir
    merged_image = cv2.warpPerspective(image2, homography_matrix, (image1.shape[1] + image2.shape[1], image2.shape[0]))
    merged_image[0:image1.shape[0], 0:image1.shape[1]] = image1

    return merged_image


# Görüntülerin yollarını belirtin
image1_path = 'C:/Users/MSI/Desktop/images/1.jpg'
image2_path = 'C:/Users/MSI/Desktop/images/2.jpg'

# Görüntüleri yükleyin
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Görüntüleri birleştirin
merged_image = image_stitching(image1, image2)

# Tek Tek Görseller
cv2.imwrite('1.jpg', image1)
cv2.imshow('1- Görüntü', image1)

cv2.imwrite('2.jpg', image2)
cv2.imshow('2- Görüntü', image2)


# Birleştirilmiş görüntüyü kaydedin veya gösterin
cv2.imwrite('birleştirilmiş_görüntü.jpg', merged_image)
cv2.imshow('Birleştirilmiş Görüntü', merged_image)

cv2.waitKey(0)
cv2.destroyAllWindows()