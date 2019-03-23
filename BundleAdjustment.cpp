#include "BundleAdjustment.h"

#include <stdio.h>
#include <string>
#include <iostream>

BundleAdjustment::BundleAdjustment()
{
	
}
BundleAdjustment::BundleAdjustment(std::vector<std::vector<MatchPoint>> leftMatchPoint,
	std::vector<std::vector<MatchPoint>> RightMatchPoint)
{
	_LeftPoint = leftMatchPoint;
	_RightPoint = RightMatchPoint;
}

BundleAdjustment::~BundleAdjustment()
{
	delete[] _ParametersInL;
	delete[] _ParametersExL;
	delete[] _3DPointsL;
	delete[] _2DPointsL;
	delete[] _ParametersInR;
	delete[] _ParametersExR;
	delete[] _3DPointsR;
	delete[] _2DPointsR;
	delete[] _Ralative;
}

void BundleAdjustment::getInitialValue()
{
	
	InitialValueEstimate initialL(_LeftPoint);
	initialL.EntryInitialValue();

	std::vector<std::vector<CorrespondingPoint>> imageMatchPointL;
	cv::Mat intrinsicMatrix;
	std::vector<std::vector<double>> extrinsicMatrix;
	imageMatchPointL = initialL.getPoints();
	intrinsicMatrix = initialL.getIntrins();
	extrinsicMatrix = initialL.getExtrinsic();

	_NumberObversations = imageMatchPointL.size();
	_NumberPoints = imageMatchPointL[0].size();
	_ParametersInL = new double[8];
	_ParametersExL = new double[6 * _NumberObversations];
	_3DPointsL = new double[_NumberObversations*_NumberPoints * 3];
	_2DPointsL = new double[_NumberObversations*_NumberPoints * 2];

	_ParametersInL[0] = intrinsicMatrix.at<double>(0, 0); //fx
	_ParametersInL[1] = intrinsicMatrix.at<double>(1, 1);//fy
	_ParametersInL[2] = intrinsicMatrix.at<double>(0, 2);//cx
	_ParametersInL[3] = intrinsicMatrix.at<double>(1, 2);//cy
	_ParametersInL[4] = 0;//k1
	_ParametersInL[5] = 0.;//k2
	_ParametersInL[6] = 0.;//p1
	_ParametersInL[7] = 0.;//p2
	for (int i = 0; i < _NumberObversations; ++i)
	{
		_ParametersExL[i*6] = extrinsicMatrix[i][0];
		_ParametersExL[i*6 + 1] = extrinsicMatrix[i][1];
		_ParametersExL[i*6 + 2] = extrinsicMatrix[i][2];
		_ParametersExL[i*6 + 3] = extrinsicMatrix[i][3];
		_ParametersExL[i*6 + 4] = extrinsicMatrix[i][4];
		_ParametersExL[i*6 + 5] = extrinsicMatrix[i][5];
		
		for (int j = 0; j < _NumberPoints; ++j)
		{
			_3DPointsL[i * _NumberPoints*3 + j * 3] = imageMatchPointL[i][j].objpoint.x;
			_3DPointsL[i * _NumberPoints*3 + j * 3 + 1] = imageMatchPointL[i][j].objpoint.y;
			_3DPointsL[i * _NumberPoints*3 + j * 3 + 2] = 0;
			_2DPointsL[i * _NumberPoints*2 + j * 2] = imageMatchPointL[i][j].imgpoint.x;
			_2DPointsL[i * _NumberPoints*2 + j * 2 + 1] = imageMatchPointL[i][j].imgpoint.y;
		}
	}
	


	InitialValueEstimate initialR(_RightPoint);
	initialR.EntryInitialValue();
	std::vector<std::vector<CorrespondingPoint>> imageMatchPointR;
	cv::Mat intrinsicMatrix_;
	std::vector<std::vector<double>> extrinsicMatrix_;
	imageMatchPointR = initialR.getPoints();
	intrinsicMatrix_ = initialR.getIntrins();
	extrinsicMatrix_ = initialR.getExtrinsic();
	_NumberObversations = imageMatchPointR.size();
	_NumberPoints = imageMatchPointR[0].size();
	_ParametersInR = new double[8];
	_ParametersExR = new double[6 * _NumberObversations];
	_3DPointsR = new double[_NumberObversations*_NumberPoints * 3];
	_2DPointsR = new double[_NumberObversations*_NumberPoints * 2];

	_ParametersInR[0] = intrinsicMatrix_.at<double>(0, 0);
	_ParametersInR[1] = intrinsicMatrix_.at<double>(1, 1);
	_ParametersInR[2] = intrinsicMatrix_.at<double>(0, 2);
	_ParametersInR[3] = intrinsicMatrix_.at<double>(1, 2);
	_ParametersInR[4] = 0;
	_ParametersInR[5] = 0.;
	_ParametersInR[6] = 0.;
	_ParametersInR[7] = 0.;
	for (int i = 0; i < _NumberObversations; ++i)
	{
		_ParametersExR[i * 6] = extrinsicMatrix_[i][0];
		_ParametersExR[i * 6 + 1] = extrinsicMatrix_[i][1];
		_ParametersExR[i * 6 + 2] = extrinsicMatrix_[i][2];
		_ParametersExR[i * 6 + 3] = extrinsicMatrix_[i][3];
		_ParametersExR[i * 6 + 4] = extrinsicMatrix_[i][4];
		_ParametersExR[i * 6 + 5] = extrinsicMatrix_[i][5];

		for (int j = 0; j < _NumberPoints; ++j)
		{
			_3DPointsR[i * _NumberPoints * 3 + j * 3] = imageMatchPointR[i][j].objpoint.x;
			_3DPointsR[i * _NumberPoints * 3 + j * 3 + 1] = imageMatchPointR[i][j].objpoint.y;
			_3DPointsR[i * _NumberPoints * 3 + j * 3 + 2] = 0;
			_2DPointsR[i * _NumberPoints * 2 + j * 2] = imageMatchPointR[i][j].imgpoint.x;
			_2DPointsR[i * _NumberPoints * 2 + j * 2 + 1] = imageMatchPointR[i][j].imgpoint.y;
		}
	}

	cv::Mat LRotate, RRotate;
	cv::Mat LRotate_vector(3, 1, CV_64F, cv::Scalar(0)), RRotate_vector(3, 1, CV_64F, cv::Scalar(0)),
		Ltranslate(3, 1, CV_64F, cv::Scalar(0)), Rtranslate(3, 1, CV_64F, cv::Scalar(0));
	LRotate_vector.at<double>(0, 0) = extrinsicMatrix[1][0];
	LRotate_vector.at<double>(1, 0) = extrinsicMatrix[1][1];
	LRotate_vector.at<double>(2, 0) = extrinsicMatrix[1][2];
	RRotate_vector.at<double>(0, 0) = extrinsicMatrix_[1][0];
	RRotate_vector.at<double>(1, 0) = extrinsicMatrix_[1][1];
	RRotate_vector.at<double>(2, 0) = extrinsicMatrix_[1][2];
	Ltranslate.at<double>(0, 0) = extrinsicMatrix[1][3];
	Ltranslate.at<double>(1, 0) = extrinsicMatrix[1][4];
	Ltranslate.at<double>(2, 0) = extrinsicMatrix[1][5];
	Rtranslate.at<double>(0, 0) = extrinsicMatrix_[1][3];
	Rtranslate.at<double>(1, 0) = extrinsicMatrix_[1][4];
	Rtranslate.at<double>(2, 0) = extrinsicMatrix_[1][5];
	cv::Rodrigues(LRotate_vector,LRotate);
	cv::Rodrigues(RRotate_vector, RRotate);
	cv::Mat RotateLtoR = LRotate * RRotate.inv();
	_translateLtoR = Ltranslate - LRotate * RRotate.inv()*Rtranslate;
	cv::Rodrigues(RotateLtoR, _RotateLtoR);
	
	_Ralative = new double[6];
	_Ralative[0] = _RotateLtoR.at<double>(0, 0);
	_Ralative[1] = _RotateLtoR.at<double>(1, 0);
	_Ralative[2] = _RotateLtoR.at<double>(2, 0);
	_Ralative[3] = _translateLtoR.at<double>(0, 0);
	_Ralative[4] = _translateLtoR.at<double>(1, 0);
	_Ralative[5] = _translateLtoR.at<double>(2, 0);
}

void BundleAdjustment::Optimize()
{
	ceres::Problem problem;
	for (int i = 0; i < _NumberObversations; ++i)
	{
		for (int j = 0; j < _NumberPoints; ++j)
		{
			// Each Residual block takes a point and a camera as input and outputs a 2
			// dimensional residual. Internally, the cost function stores the observed
			// image location and compares the reprojection against the observation.
			ceres::CostFunction* cost_function =
				SnavelyReprojectionError::Create(_2DPointsL[_NumberPoints * i*2 + j * 2],
					_2DPointsL[_NumberPoints * i*2 + j * 2 + 1], _3DPointsL[_NumberPoints * i*3 + j * 3],
					_3DPointsL[_NumberPoints * i*3 + j * 3 + 1],
					_3DPointsL[_NumberPoints * i*3 + j * 3 + 2],
					_2DPointsR[_NumberPoints * i * 2 + j * 2],
					_2DPointsR[_NumberPoints * i * 2 + j * 2 + 1]);
			problem.AddResidualBlock(cost_function,NULL /* squared loss */, _ParametersInL,_ParametersExL + i *6,_Ralative,_ParametersInR);
		}
		
	}

	// Make Ceres automatically detect the bundle structure. Note that the
	// standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
	// for standard bundle adjustment problems.
	ceres::Solver::Options options;
	options.max_num_iterations = 50;
	options.num_threads = 4;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
}

void BundleAdjustment::SaveResult(const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << "LeftCamera" << "{";
	fs << "fx" << _ParametersInL[0];
	fs << "fy" << _ParametersInL[1];
	fs << "cx" << _ParametersInL[2];
	fs << "cy" << _ParametersInL[3];
	fs << "k1" << _ParametersInL[4];
	fs << "k2" << _ParametersInL[5];
	fs << "p1" << _ParametersInL[6];
	fs << "p2" << _ParametersInL[7];
	fs << "}";
	fs << "RightCamera" << "{";
	fs << "fx" << _ParametersInR[0];
	fs << "fy" << _ParametersInR[1];
	fs << "cx" << _ParametersInR[2];
	fs << "cy" << _ParametersInR[3];
	fs << "k1" << _ParametersInR[4];
	fs << "k2" << _ParametersInR[5];
	fs << "p1" << _ParametersInR[6];
	fs << "p2" << _ParametersInR[7];
	fs << "}";
	fs << "Left2Right" << "{";
	fs << "rotate";
	fs << "[";
	fs<< _Ralative[0] << _Ralative[1] << _Ralative[2];
	fs << "]";
	fs << "translate";
	fs << "[";
	fs << _Ralative[3] << _Ralative[4] << _Ralative[5];
	fs << "]";
	fs << "}";
	fs.release();
}
