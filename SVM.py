from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

# Training 3 SVM models for RGB, HSV, and grayscale color spaces
# Using a Gaussian kernel for non-linear classification
RGB_SVM = SVC(kernel="rbf").fit(X_train_rgb, y_train)
HSV_SVM = SVC(kernel="rbf").fit(X_train_hsv, y_train)
GRAY_SVM = SVC(kernel="rbf").fit(X_train_gray, y_train)

# Predicting on the test set
RGB_SVM_pred = RGB_SVM.predict(X_test_rgb)
HSV_SVM_pred = HSV_SVM.predict(X_test_hsv)
GRAY_SVM_pred = GRAY_SVM.predict(X_test_gray)


# Evaluating the SVM models
print("RGB SVM Classification Report:")
print(classification_report(y_test, RGB_SVM_pred))
print(f"RGB Accuracy: {RGB_SVM.score(X_test_rgb, y_test):.4f}")

print("HSV SVM Classification Report:")
print(classification_report(y_test, HSV_SVM_pred))
print(f"HSV Accuracy: {HSV_SVM.score(X_test_hsv, y_test):.4f}")

print("Grayscale SVM Classification Report:")
print(classification_report(y_test, GRAY_SVM_pred))
print(f"Grayscale Accuracy: {GRAY_SVM.score(X_test_gray, y_test):.4f}")


# Confusion matrices
print("RGB SVM Confusion Matrix:")
print(confusion_matrix(y_test, RGB_SVM_pred))

print("HSV SVM Confusion Matrix:")
print(confusion_matrix(y_test, HSV_SVM_pred))

print("Grayscale SVM Confusion Matrix:")
print(confusion_matrix(y_test, GRAY_SVM_pred))
