#include "CalibPanelDetector.h"
#include <cmath>

CalibPanelDetector::CalibPanelDetector(std::string path)
{
	_SrcImage = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	_ShowImage = _SrcImage.clone();
	_vCalibPanel3DPoints = cv::Mat(CALIBPANEL_NUMBERS_Y, CALIBPANEL_NUMBERS_X, CV_64FC2);
}

CalibPanelDetector::CalibPanelDetector()
{

}

CalibPanelDetector::~CalibPanelDetector()
{

}

void CalibPanelDetector::EntryEllipseDetector()
{
	Processing();
	Build3DPointsForCalibPanel();
	HomographyFromFivePoints();
	
}

void CalibPanelDetector::Processing()
{
	cv::Mat blur_Image, canny_Image;
	//guassian blur for src image
	cv::GaussianBlur(_SrcImage, blur_Image, cv::Size(7, 7), 1.0, 1.0, cv::BORDER_REPLICATE);
	//elemate some nosie which may attach fitting  
	cv::Mat element(10, 10, CV_8U, cv::Scalar(0));
	cv::morphologyEx(blur_Image, blur_Image, cv::MORPH_CLOSE, element);
	// detecte edge image using canny methond
	Canny(blur_Image, canny_Image, 70, 2 * 70, 3);

	//find every contours in image and judge whether is ellipse or not
	std::vector<std::vector<cv::Point> > contours;
	findContours(canny_Image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);	

	// location five ellipse 
	std::vector<cv::RotatedRect>& vEllipseBoxs = _vEllipseBoxs;
	
	int count = 0;
	double size = 0;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		if (contours[i].size() > 5)
		{
			cv::RotatedRect ellipseBox = cv::fitEllipse(contours[i]);
			vEllipseBoxs.push_back(ellipseBox);

			double area = cv::contourArea(contours[i]);
			double length = cv::arcLength(contours[i], false);

			if (circularityCriteria1(length, area))
			{
				size += length;
				count++;
			}
		}
		
	}
	_AverageLength = size / count;

	const double MIN_RATIO_BIGPNT = 1.8;
	const double MAX_RATIO_BIGPNT = 2.5;


	std::vector<int> indexs;
	//int indexs[5] = { 0 };
	int nCountBigpoints = 0;
	for (size_t i = 0; i < contours.size(); ++i)
	{
		double perimeter = cv::arcLength(contours[i], false);
		double area = cv::contourArea(contours[i]);
		if (circularityCriteria1(perimeter, area))
		{
			if (perimeter > MIN_RATIO_BIGPNT*_AverageLength
				&& perimeter < MAX_RATIO_BIGPNT*_AverageLength)
			{
				indexs.push_back(i); //记录5个大点在contours中的索引
				nCountBigpoints++;
			}

		}
		
	}

	if (nCountBigpoints != 5)
	{
		
	}

	std::vector<cv::Point2d> FiveBigCenter;
	for (int i = 0; i < nCountBigpoints; ++i)
	{
		FiveBigCenter.push_back(vEllipseBoxs[indexs[i]].center);
	}

	double minDistance = std::numeric_limits<float>::max();

	int id22, id47, id51, id56, id76;
	//search id47 and id56 using min and max distance
	for (size_t i = 0; i < FiveBigCenter.size(); ++i)
	{
		cv::Point2d p1 = FiveBigCenter[i];
		for (size_t j = i + 1; j < FiveBigCenter.size(); ++j)
		{
			cv::Point2d p2 = FiveBigCenter[j];
			double distance = computer_distance(p1, p2);
			if (distance < minDistance)
			{
				id47 = i;
				id56 = j;
				minDistance = distance;
			}
		}

	}
	std::vector<cv::Point2d> RestBigCenter;
	std::vector<int> three_index;
	for (size_t i = 0; i < 5; ++i)
	{
		if (i != id47 && i != id56)
		{
			RestBigCenter.push_back(FiveBigCenter[i]);
			three_index.push_back(i);
		}
	}

	cv::Point2d pv0, pvi[3], pvj[3];
	pointVecFromTwoPoints(FiveBigCenter[id47], FiveBigCenter[id56], pv0);
	pointVecFromTwoPoints(FiveBigCenter[id47], RestBigCenter[0], pvi[0]);
	pointVecFromTwoPoints(FiveBigCenter[id47], RestBigCenter[1], pvi[1]);
	pointVecFromTwoPoints(FiveBigCenter[id47], RestBigCenter[2], pvi[2]);
	double c1 = pv0.cross(pvi[1]);
	if (c1 > 0)
	{
		int temp = id47;
		id47 = id56;
		id56 = temp;
	}

	pointVecFromTwoPoints(FiveBigCenter[id47], RestBigCenter[0], pvi[0]);
	pointVecFromTwoPoints(FiveBigCenter[id47], RestBigCenter[1], pvi[1]);
	pointVecFromTwoPoints(FiveBigCenter[id47], RestBigCenter[2], pvi[2]);
	pointVecFromTwoPoints(FiveBigCenter[id56], RestBigCenter[0], pvj[0]);
	pointVecFromTwoPoints(FiveBigCenter[id56], RestBigCenter[1], pvj[1]);
	pointVecFromTwoPoints(FiveBigCenter[id56], RestBigCenter[2], pvj[2]);

	double distance_47_56 = cv::norm(pv0);
	double distance[6];
	for (int i = 0; i < 3; ++i)
	{
		double d1 = cv::norm(pvi[i]);
		distance[i*2] = d1;
		double d2 = cv::norm(pvj[i]);
		distance[i*2 + 1] = d2;
	}
	double dot[3];
	for (int i = 0; i < 3; ++i)
	{
		double theta = (distance_47_56*distance_47_56 + distance[2 * i] * distance[i * 2] - distance[2 * i + 1] * distance[2 * i + 1])
			/ (2 * distance_47_56 * distance[2 * i]);
		dot[i] = theta;
	}
	double maxdot = std::numeric_limits<float>::min();
	double mindot = std::numeric_limits<float>::max();
	for (int i = 0; i < 3; ++i)
	{
		if (dot[i] > maxdot)
		{
			id76 = three_index[i];
			maxdot = dot[i];
		}
		if (dot[i] < mindot)
		{
			id22 = three_index[i];
			mindot = dot[i];
		}
	}

	for (int i = 0; i < 5; ++i)
	{
		if (i == id47 || i == id56 || i == id22 || i == id76)
			continue;
		id51 = i;
	}
	
	_FiveBigCenter.push_back(FiveBigCenter[id22]);
	_FiveBigCenter.push_back(FiveBigCenter[id47]);
	_FiveBigCenter.push_back(FiveBigCenter[id51]);
	_FiveBigCenter.push_back(FiveBigCenter[id56]);
	_FiveBigCenter.push_back(FiveBigCenter[id76]);

	/*std::string ID[5] = { "id22" ,"id47","id51","id56","id76" };
	for (size_t i = 0; i < _FiveBigCenter.size(); ++i)
	{
		cv::putText(_SrcImage, ID[i], _FiveBigCenter[i], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2, 8);
	}*/
	
}

void CalibPanelDetector::Build3DPointsForCalibPanel()
{
	for (int r = 0; r < CALIBPANEL_NUMBERS_Y; r++)
	{
		cv::Point2d* data = _vCalibPanel3DPoints.ptr<cv::Point2d>(r);
		for (int c = 0; c < CALIBPANEL_NUMBERS_X; c++)
		{
			data[c].x = c * CALIBPANEL_SPACING_X;
			data[c].y = r * CALIBPANEL_SPACING_Y;
		}
	}
}

void CalibPanelDetector::HomographyFromFivePoints()
{
	cv::Mat FiveBig3DPoints(5, 1, CV_64FC2);
	FiveBig3DPoints.at<cv::Point2d>(0, 0) = _vCalibPanel3DPoints.at<cv::Point2d>(2, 4);//2*9+4
	FiveBig3DPoints.at<cv::Point2d>(1, 0) = _vCalibPanel3DPoints.at<cv::Point2d>(5, 2);//5*9+2
	FiveBig3DPoints.at<cv::Point2d>(2, 0) = _vCalibPanel3DPoints.at<cv::Point2d>(5, 6);//5*9+6
	FiveBig3DPoints.at<cv::Point2d>(3, 0) = _vCalibPanel3DPoints.at<cv::Point2d>(6, 2);//6*9+2
	FiveBig3DPoints.at<cv::Point2d>(4, 0) = _vCalibPanel3DPoints.at<cv::Point2d>(8, 4);//6*9+2

	cv::Mat FiveBig2DPoints(5, 1, CV_64FC2);
	for (size_t i = 0; i < _FiveBigCenter.size(); ++i)
	{
		FiveBig2DPoints.at<cv::Point2d>(i, 0) = _FiveBigCenter[i];
	}
	cv::Mat Homography = cv::findHomography(FiveBig3DPoints, FiveBig2DPoints, CV_RANSAC);

	cv::Mat P2d;
	cv::perspectiveTransform(_vCalibPanel3DPoints, P2d, Homography);
	MatchPoint corrsPoint;
	for (int i = 0; i < _vCalibPanel3DPoints.rows; ++i)
	{
		for (int j = 0; j < _vCalibPanel3DPoints.cols; ++j)
		{
			for (size_t k = 0; k < _vEllipseBoxs.size(); ++k)
			{
				if ((P2d.at<cv::Point2d>(i, j).x - _AverageLength < _vEllipseBoxs[k].center.x) &&
					(P2d.at<cv::Point2d>(i, j).x + _AverageLength > _vEllipseBoxs[k].center.x))
				{
					if ((P2d.at<cv::Point2d>(i, j).y - _AverageLength < _vEllipseBoxs[k].center.y) &&
						(P2d.at<cv::Point2d>(i, j).y + _AverageLength > _vEllipseBoxs[k].center.y))
					{
						std::string ID = std::to_string(i * 9 + j);
						cv::putText(_ShowImage, ID, _vEllipseBoxs[k].center, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 2, 8);
						corrsPoint.identifier = i * 9 + j;
						corrsPoint.imgpoint = _vEllipseBoxs[k].center;
						corrsPoint.objpoint = cv::Point3d(_vCalibPanel3DPoints.at<cv::Point2d>(i, j).x, _vCalibPanel3DPoints.at<cv::Point2d>(i, j).y, 0);
						_vCalibPanelMatchPoints.push_back(corrsPoint);
					}
				}

			}

		}

	}
}

void CalibPanelDetector::ShowResult()
{
	cv::Mat show_image;
	cv::imwrite("result.png", _ShowImage);
	cv::resize(_ShowImage, show_image, cv::Size(_ShowImage.rows / 2, _ShowImage.cols / 2));
	cv::imshow("Main", show_image);
	cv::waitKey(20);
	


}



double CalibPanelDetector::computer_distance(cv::Point2f p1, cv::Point2f p2)
{
	cv::Vec2f v1(p1.x - p2.x, p1.y - p2.y);
	double distance = cv::norm(v1);

	return distance;
}

void CalibPanelDetector::SaveResult(std::string& path)
{
	std::vector<std::string> _fileNames;
	getAllFiles(path, _fileNames);
	int file_number = _fileNames.size();
	std::string ordeing = std::to_string(file_number);
	std::string separator = "\\";
	std::string format = ".txt";
	std::ofstream dataFile;
	dataFile.open(path + separator + "image" + ordeing + format, std::ofstream::out);
	for (size_t i = 0; i < _vCalibPanelMatchPoints.size(); ++i)
	{
		dataFile << std::setprecision(10) << _vCalibPanelMatchPoints[i].imgpoint.x << " " << _vCalibPanelMatchPoints[i].imgpoint.y << " "
			<< _vCalibPanelMatchPoints[i].objpoint.x << " " << _vCalibPanelMatchPoints[i].objpoint.y << std::endl;
	}
	dataFile.close();
}
std::vector<MatchPoint> CalibPanelDetector::getMatchPoint() const
{
	return _vCalibPanelMatchPoints;
}

