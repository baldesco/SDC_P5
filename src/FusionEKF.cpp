#include "FusionEKF.h"
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;
using std::pow;
using std::cos;
using std::sin;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  //measurement matrix - laser
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;

  //The measurement matrix for the radar is left empty (zeros).

  // Initialize elements of the KalmanFilter object ekf ---------------------------

  // 4D state vector, we don't know yet the values of the x state
  VectorXd x = VectorXd(4);
  // Covariance matrix P
  MatrixXd P = MatrixXd(4, 4);
  P	<<	1,	0,	0,	 0,
				0,	1,	0,	 0,
				0,	0, 1000, 0,
				0,	0, 0,	1000;
  // Transition matrix F
  MatrixXd F = MatrixXd(4, 4);
  F << 1, 0, 1, 0,
       0, 1, 0, 1,
       0, 0, 1, 0,
       0, 0, 0, 1;
  // Process noise covariance matrix Q - initialized empty
  MatrixXd Q_ = MatrixXd(4, 4);
  Q_ <<	0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 0, 0;
  /**
   * Initialize the kalman filter object:
   * The laser H and R matrices are used here to initialize ekf_,
   * but this will depend on the sensor that does the measurement.
   * */
  ekf_.Init(x, P, F, H_laser_, R_laser_, Q_);

  // set the acceleration noise components
  noise_ax = 9;
  noise_ay = 9;

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // get the measurement values (polar coordinates)
      float ro = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float ro_dot = measurement_pack.raw_measurements_[2];
      // Convert the values to cartesian coordinates
      float px = ro * cos(phi);
      float py = ro * sin(phi);

      // Initialize state
      ekf_.x_ << px, py, 0.0, 0.0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state directly with the measurement values
      ekf_.x_ <<  measurement_pack.raw_measurements_[0],
                  measurement_pack.raw_measurements_[1],
                  0.0,
                  0.0;
    }

    // done initializing, no need to predict or update
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  // update the time of the last measurement
  previous_timestamp_ = measurement_pack.timestamp_;

  /**
   * Prediction
   */

  // Modify the F matrix so that the time is integrated
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;
  // Set the process covariance matrix Q
  ekf_.Q_(0,0) = pow(dt,4)/4 * noise_ax;
  ekf_.Q_(0,2) = pow(dt,3)/2 * noise_ax;
  ekf_.Q_(1,1) = pow(dt,4)/4 * noise_ay;
  ekf_.Q_(1,3) = pow(dt,3)/2 * noise_ay;
  ekf_.Q_(2,0) = pow(dt,3)/2 * noise_ax;
  ekf_.Q_(2,2) = pow(dt,2) * noise_ax;
  ekf_.Q_(3,1) = pow(dt,3)/2 * noise_ay;
  ekf_.Q_(3,3) = pow(dt,2) * noise_ay;
  // Call the prediction function
  ekf_.Predict();
  /**
   * Update
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
//   cout << "x_ = " << ekf_.x_ << endl;
//   cout << "P_ = " << ekf_.P_ << endl;
}
