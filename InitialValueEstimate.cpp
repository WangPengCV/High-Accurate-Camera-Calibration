#include "InitialValueEstimate.h"
#include <fstream>
#include <io.h>
#include <Eigen\Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>

InitialValueEstimate::InitialValueEstimate()
{

}

InitialValueEstimate::~InitialValueEstimate()
{

}

InitialValueEstimate::InitialValueEstimate(std::string& path)
{
	getAllFiles(path, _fileNames);
	_imageNumber = _fileNames.size();
	_imageMatchPoint.reserve(_imageNumber);
	for (size_t i = 0; i < _fileNames.size(); ++i)
	{
		std::ifstream readFile;
		std::string separator = "\\";
		//std::string format = ".txt";

		
		std::string feature;
		readFile.open(path + separator + _fileNames[i], std::ios::in);
		CorrespondingPoint OnePoint;
		std::vector<CorrespondingPoint> OneimagePoint;
		OneimagePoint.clear();
		while (getline(readFile, feature))
		{
			std::stringstream stringin(feature);
			stringin >> OnePoint.imgpoint.x;
			stringin >> OnePoint.imgpoint.y;
			stringin >> OnePoint.objpoint.x;
			stringin >> OnePoint.objpoint.y;
			OneimagePoint.push_back(OnePoint);
		}
		readFile.close();
		_imageMatchPoint.push_back(OneimagePoint);
	}
	

}
InitialValueEstimate::InitialValueEstimate(std::vector<std::vector<MatchPoint>> CalibPoint)
{
	std::vector<CorrespondingPoint> OneimagePoint;
	_imageMatchPoint.reserve(CalibPoint.size());
	_imageNumber = CalibPoint.size();
	CorrespondingPoint OnePoint;
	for (size_t i = 0; i < CalibPoint.size(); ++i)
	{
		OneimagePoint.clear();
		for (size_t j = 0; j < CalibPoint[i].size(); ++j)
		{
			OnePoint.imgpoint.x = CalibPoint[i][j].imgpoint.x;
			OnePoint.imgpoint.y = CalibPoint[i][j].imgpoint.y;
			OnePoint.objpoint.x = CalibPoint[i][j].objpoint.x;
			OnePoint.objpoint.y = CalibPoint[i][j].objpoint.y;
			OneimagePoint.push_back(OnePoint);
		}
		_imageMatchPoint.push_back(OneimagePoint);
	}
	
}


void InitialValueEstimate::EntryInitialValue()
{
	HomographyFromMatchpoint();
	EstimateParameters();

}

void InitialValueEstimate::HomographyFromMatchpoint()
{
	_Homography.reserve(_imageNumber);
	for (int n = 0; n < _imageNumber; ++n)
	{
		cv::Mat CalibPanel3dPoint(CALIBPANEL_NUMBERS_Y*CALIBPANEL_NUMBERS_X,1, CV_32FC2, cv::Scalar(0));
		for (int i = 0; i < _imageMatchPoint[n].size(); ++i)
		{
			CalibPanel3dPoint.at<cv::Point2f>(i, 0) = _imageMatchPoint[n][i].objpoint;
		}
		cv::Mat CalibPanel2dPoint(CALIBPANEL_NUMBERS_Y*CALIBPANEL_NUMBERS_X,1, CV_32FC2, cv::Scalar(0));
		
		for (int i = 0; i < _imageMatchPoint[n].size(); ++i)
		{
			CalibPanel2dPoint.at<cv::Point2f>(i, 0) = _imageMatchPoint[n][i].imgpoint;
		}
		cv::Mat H = cv::findHomography(CalibPanel3dPoint, CalibPanel2dPoint, CV_RANSAC);
		_Homography.push_back(H);
	}
	
}

void InitialValueEstimate::EstimateParameters()
{
	cv::Mat A(_imageNumber * 2, 6, CV_64FC1, cv::Scalar(0));
	cv::Mat b(6, 1, CV_64FC1, cv::Scalar(0));
	for (int i = 0; i < _imageNumber; ++i)
	{
		cv::Mat H = _Homography[i];
		A.at<double>(i * 2, 0) = H.at<double>(0, 0)*H.at<double>(0, 1);
		A.at<double>(i * 2, 1) = H.at<double>(0, 0)*H.at<double>(1, 1) + H.at<double>(1, 0)*H.at<double>(0, 1);
		A.at<double>(i * 2, 2) = H.at<double>(1, 0)*H.at<double>(1, 1);
		A.at<double>(i * 2, 3) = H.at<double>(2, 0)*H.at<double>(0, 1) + H.at<double>(0, 0)*H.at<double>(2, 1);
		A.at<double>(i * 2, 4) = H.at<double>(2, 0)*H.at<double>(1, 1) + H.at<double>(1, 0)*H.at<double>(2, 1);
		A.at<double>(i * 2, 5) = H.at<double>(2, 0)*H.at<double>(2, 1);
		A.at<double>(i * 2 + 1, 0) = H.at<double>(0, 0)*H.at<double>(0, 0) - (H.at<double>(0, 1)*H.at<double>(0, 1));
		A.at<double>(i * 2 + 1, 1) = H.at<double>(0, 0)*H.at<double>(1, 0) + H.at<double>(1, 0)*H.at<double>(0, 0) - 
			(H.at<double>(0, 1)*H.at<double>(1, 1) + H.at<double>(1, 1)*H.at<double>(0, 1));
		A.at<double>(i * 2 + 1, 2) = H.at<double>(1, 0)*H.at<double>(1, 0) - H.at<double>(1, 1)*H.at<double>(1, 1);
		A.at<double>(i * 2 + 1, 3) = H.at<double>(2, 0)*H.at<double>(0, 0) + H.at<double>(0, 0)*H.at<double>(2, 0) - 
			(H.at<double>(2, 1)*H.at<double>(0, 1) + H.at<double>(0, 1)*H.at<double>(2, 1));
		A.at<double>(i * 2 + 1, 4) = H.at<double>(2, 0)*H.at<double>(1, 0) + H.at<double>(1, 0)*H.at<double>(2, 0) - 
			(H.at<double>(2, 1)*H.at<double>(1, 1) + H.at<double>(1, 1)*H.at<double>(2, 1));
		A.at<double>(i * 2 + 1, 5) = H.at<double>(2, 0)*H.at<double>(2, 0) - H.at<double>(2, 1)*H.at<double>(2, 1);
	}

	cv::Mat U, W, VT;
	cv::SVD::compute(A, W, U,VT);
	b = VT.rowRange(VT.rows-1, VT.rows).clone();
	cv::Mat x;
	cv::transpose(b, b);
	x = A * b;
	Eigen::MatrixXd B(3, 3);
	B << b.at<double>(0, 0), b.at<double>(1, 0), b.at<double>(3, 0), b.at<double>(1, 0),
		b.at<double>(2, 0), b.at<double>(4, 0), b.at<double>(3, 0), b.at<double>(4, 0), b.at<double>(5, 0);
	Eigen::LLT<Eigen::MatrixXd> lltOfA(B);
	Eigen::MatrixXd LT = lltOfA.matrixL();
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			LT(i, j) = LT(i, j) / LT(2, 2);
		}
	}
	Eigen::MatrixXd L = LT.transpose();
	Eigen::MatrixXd K = L.inverse();
	cv::eigen2cv(K, _intrinsicMatrix);
	_extrinsicMatrix.reserve(_imageNumber);
	for (int i = 0; i < _imageNumber; ++i)
	{
		cv::Mat H = _Homography[i];
		cv::Mat h1 = H.colRange(0,1).clone();
		cv::Mat h2 = H.colRange(1,2).clone();
		cv::Mat h3 = H.colRange(2, 3).clone();
		cv::Mat tempMat = _intrinsicMatrix.inv()*h1;
		double temp = cv::norm(tempMat,cv::NORM_L2);
		double lamda = 1.0 / temp;
		cv::Mat r1 = lamda * _intrinsicMatrix.inv()*h1;
		cv::Mat r2 = lamda * _intrinsicMatrix.inv()*h2;
		cv::Mat r3 = r1.cross(r2);
		cv::Mat t = lamda * _intrinsicMatrix.inv()*h3;
		cv::Mat Q;
		cv::hconcat(r1, r2, Q);
		cv::hconcat(Q, r3, Q);
		cv::Mat U1, W1, V1T;
		cv::SVD::compute(Q, W1, U1, V1T);
		cv::Mat R = U1 * V1T;

		cv::Mat rotate_vector;
		cv::Rodrigues(R, rotate_vector);
		std::vector<double> camera_observation(6);
		camera_observation[0] = rotate_vector.at<double>(0, 0);
		camera_observation[1] = rotate_vector.at<double>(1, 0);
		camera_observation[2] = rotate_vector.at<double>(2, 0);
		camera_observation[3] = t.at<double>(0, 0);
		camera_observation[4] = t.at<double>(1, 0);
		camera_observation[5] = t.at<double>(2, 0);
		_extrinsicMatrix.push_back(camera_observation);
	}

	
}

std::vector<std::vector<CorrespondingPoint>> InitialValueEstimate::getPoints() const
{
	return _imageMatchPoint;
}

cv::Mat InitialValueEstimate::getIntrins() const
{
	return _intrinsicMatrix;
}

std::vector<std::vector<double>>  InitialValueEstimate::getExtrinsic() const
{
	return _extrinsicMatrix;
}




	
