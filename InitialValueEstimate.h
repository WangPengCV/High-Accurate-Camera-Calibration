#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "DataType.h"


#define CALIBPANEL_NUMBERS_X 9
#define CALIBPANEL_NUMBERS_Y 11



class InitialValueEstimate
{
public:
	InitialValueEstimate();
	InitialValueEstimate(std::string& path);
	InitialValueEstimate(std::vector<std::vector<MatchPoint>> CalibPoint);
	~InitialValueEstimate();

public:
	void HomographyFromMatchpoint();
	void EstimateParameters();
	void EntryInitialValue();
	std::vector<std::vector<CorrespondingPoint>> getPoints() const ;
	cv::Mat getIntrins() const;
	std::vector<std::vector<double>> getExtrinsic() const;

private:
	std::vector<std::vector<CorrespondingPoint>> _imageMatchPoint;
	std::vector<std::string> _fileNames;
	std::vector<cv::Mat> _Homography;
	int _imageNumber;
	cv::Mat _intrinsicMatrix;
	std::vector<std::vector<double>> _extrinsicMatrix;
	
};