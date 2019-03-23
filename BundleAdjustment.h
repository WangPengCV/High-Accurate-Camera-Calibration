#pragma once
#include "InitialValueEstimate.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"



struct SnavelyReprojectionError 
{
	SnavelyReprojectionError(double observed_x_l, double observed_y_l, double point_x_l, double point_y_l, double point_z_l, double observed_x_r, double observed_y_r)
		: observed_x_l(observed_x_l), observed_y_l(observed_y_l),point_x_l(point_x_l),point_y_l(point_y_l),point_z_l(point_z_l), observed_x_r(observed_x_r), observed_y_r(observed_y_r)
	{
		//*Point = point_x;
		//Point[1] = point_y;
		//Point[2] = point_z;
	}
	template <typename T>
	bool operator()(const T* const cameraInL, const T* const cameraExL, const T* const ralativeIn,const T* const cameraInR, T* residuals) const
	{
		// camera[0,1,2] are the angle-axis rotation.
		T pL[3];
		T* pointL = new T[3];
		*pointL =  T (point_x_l);
		pointL[1] = T(point_y_l);
		pointL[2] = T(point_z_l);

		ceres::AngleAxisRotatePoint(cameraExL, pointL, pL);

		// camera[3,4,5] are the translation.
		pL[0] += cameraExL[3];
		pL[1] += cameraExL[4];
		pL[2] += cameraExL[5];

		// Compute the center of distortion. The sign change comes from
		// the camera model that Noah Snavely's Bundler assumes, whereby
		// the camera coordinate system has a negative z axis.
		T xpL = pL[0] / pL[2];
		T ypL = pL[1] / pL[2];

		// Apply second and fourth order radial distortion.
		const T& kL1 = cameraInL[4];
		const T& kL2 = cameraInL[5];
		const T& pL1 = cameraInL[6];
		const T& pL2 = cameraInL[7];
		T rL2 = xpL * xpL + ypL * ypL;
		T distortionL = (T)1.0 + rL2 * (kL1 + kL2 * rL2);

		// Compute final projected point position.
		const T& focalxL = cameraInL[0];
		const T& focalyL = cameraInL[1];
		const T& uxL = cameraInL[2];
		const T& uyL = cameraInL[3];
		T predicted_xL = focalxL * (distortionL * xpL + (T)2*pL1*xpL*ypL + pL2*(rL2+(T)2*xpL*xpL)) + uxL;
		T predicted_yL = focalyL * (distortionL * ypL + pL1*(rL2 + (T)2*ypL*ypL) + (T)2*pL2*xpL*ypL) + uyL;

		residuals[0] = predicted_xL - observed_x_l;
		residuals[1] = predicted_yL - observed_y_l;
		// camera[0,1,2] are the angle-axis rotation.
		T pR[3];
		T p[3];
		ceres::AngleAxisRotatePoint(cameraExL, pointL, p);
		p[0] += cameraExL[3];
		p[1] += cameraExL[4];
		p[2] += cameraExL[5];

		ceres::AngleAxisRotatePoint(ralativeIn, p, pR);

		// camera[3,4,5] are the translation.
		pR[0] += ralativeIn[3];
		pR[1] += ralativeIn[4];
		pR[2] += ralativeIn[5];

		// Compute the center of distortion. The sign change comes from
		// the camera model that Noah Snavely's Bundler assumes, whereby
		// the camera coordinate system has a negative z axis.
		T xpR = pR[0] / pR[2];
		T ypR = pR[1] / pR[2];

		// Apply second and fourth order radial distortion.
		const T& kR1 = cameraInR[4];
		const T& kR2 = cameraInR[5];
		const T& pR1 = cameraInR[6];
		const T& pR2 = cameraInR[7];
		T rR2 = xpR * xpR + ypR * ypR;
		T distortionR = (T)1.0 + rR2 * (kR1 + kR2 * rR2);

		// Compute final projected point position.
		const T& focalxR = cameraInR[0];
		const T& focalyR = cameraInR[1];
		const T& uxR = cameraInR[2];
		const T& uyR = cameraInR[3];
		T predicted_xR = focalxR * (distortionR * xpR + (T)2 * pR1*xpR*ypR + pR2 * (rR2 + (T)2 * xpR*xpR)) + uxR;
		T predicted_yR = focalyR * (distortionR * ypR + pR1 * (rR2 + (T)2 * ypR*ypR) + (T)2 * pR2*xpR*ypR) + uyR;



		// The error is the difference between the predicted and observed position.
		residuals[2] = predicted_xR - observed_x_r;
		residuals[3] = predicted_yR - observed_y_r;
		delete[] pointL;
		return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x_l,
		const double observed_y_l,const double point_x, const double point_y, const double point_z,
		const double observerd_x_r,const double observerd_y_r)
	{
		return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 4, 8,6,6,8>(
			new SnavelyReprojectionError(observed_x_l, observed_y_l,point_x,point_y,point_z, observerd_x_r, observerd_y_r)));
	}

	double observed_x_l;
	double observed_y_l;
	double point_x_l;
	double point_y_l;
	double point_z_l;
	double observed_x_r;
	double observed_y_r;
	//double* Point = new double[3];
};

class BundleAdjustment
{
public:
	BundleAdjustment();
	BundleAdjustment(std::vector<std::vector<MatchPoint>> leftPoint, 
		std::vector<std::vector<MatchPoint>> RightPoint);
	~BundleAdjustment();
public:
	void getInitialValue();
	void Optimize();
	void SaveResult(const std::string& filename);

private:
	std::vector<std::vector<MatchPoint>> _LeftPoint;
	std::vector<std::vector<MatchPoint>> _RightPoint;
	cv::Mat _RotateLtoR;
	cv::Mat _translateLtoR;
	int _NumberObversations;
	int _NumberPoints;
	double* _ParametersInL;
	double* _ParametersExL;
	double* _3DPointsL;
	double* _2DPointsL;
	double* _ParametersInR;
	double* _ParametersExR;
	double* _3DPointsR;
	double* _2DPointsR;
	double* _Ralative;
	
};

