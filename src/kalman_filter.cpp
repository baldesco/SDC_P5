#include "kalman_filter.h"
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::pow;
using std::atan2;
using std::acos;
using std::cout;
using std::endl;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // Apply the prediction equations
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Create variables needed for the update step
  int x_size = x_.size();
  MatrixXd Ht = H_.transpose();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  VectorXd z_pred = H_ * x_;
  // Update calculations
  VectorXd y = z - z_pred;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  //new estimate
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // get the state parameters (cartesian coordinates)
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  // Calculate the state parameters (polar coords) using nonlinear function h(x)
  double ro, phi, ro_dot;
  ro = pow(px*px + py*py, 0.5);
  phi = atan2(py, px);

  if (ro <= 0.0001){
    cout << "UpdateEKF () - Warning - Very small value for the rho variable prediction." << endl;
    ro_dot = (px*vx + py*vy) / 0.0001;
  } else {
    ro_dot = (px*vx + py*vy) / ro;
  }
  // Create the vector z_pred with ro, phi and ro_dot
  VectorXd z_pred(3);
  z_pred << ro, phi, ro_dot;
  // Calculate the error vector, and ensure that its phi value is between the range -PI and PI
  VectorXd y = z - z_pred;
  const float  PI_F = acos(-1); // the inverse cosine of -1 radians equals PI
  while(y[1] > PI_F){
    y[1] -= 2*PI_F;
  }
  while(y[1] < -PI_F){
    y[1] += 2*PI_F;
  }

  // Create variables needed for the update step
  int x_size = x_.size();
  MatrixXd Ht = H_.transpose();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  // Update calculations
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  //new estimate
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}
