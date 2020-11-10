/*
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>

//Image
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include "Hungarian.h"

extern double iouThreshold;
extern int min_hits;
extern int max_age;
extern std::vector<KalmanTracker> trackers;
extern std::vector<std::vector<TrackingBox>> detFrameData;


std::vector<cv::Rect_<float>> get_predictions() {
	std::vector<cv::Rect_<float>> predBoxes;
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		cv::Rect_<float> pBox = (*it).predict();
		//std::cout << pBox.x << " " << pBox.y << std::endl;
		if (pBox.x >= 0 && pBox.y >= 0)
		{
			predBoxes.push_back(pBox);
			it++;
		}
		else
		{
			it = trackers.erase(it);
			//cerr << "Box invalid at frame: " << frame_count << endl;
		}
	}
	return predBoxes;
}

double GetIOU(cv::Rect_<float> bb_dr, cv::Rect_<float> bb_gt) {
	float in = (bb_dr & bb_gt).area();
	float un = bb_dr.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	double iou = in / un;

	return iou;
}


MatchItems Sort_match(std::vector<std::vector<TrackingBox>> detFrameData, int __, std::vector<cv::Rect_<float>> predictedBoxes) {
	int f_num = detFrameData.size() - 1;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;
	trkNum = predictedBoxes.size();
	detNum = detFrameData[f_num].size();

	std::vector<std::vector<double>> iouMatrix;
	std::vector<int> assignment;

	std::set<int> unmatchedDetections;
	std::set<int> unmatchedTrajectories;
	std::set<int> allItems;
	std::set<int> matchedItems;
	// result
	std::vector<cv::Point> matchedPairs;
	MatchItems matched_result;

	iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));


	for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
	{
		for (unsigned int j = 0; j < detNum; j++)
		{
			// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
			iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[f_num][j].box);
		}
	}

	// solve the assignment problem using hungarian algorithm.
	// the resulting assignment is [track(prediction) : detection], with len=preNum
	HungarianAlgorithm HungAlgo;
	HungAlgo.Solve(iouMatrix, assignment);

	// find matches, unmatched_detections and unmatched_predictions
	if (detNum > trkNum) //	there are unmatched detections
	{
		for (unsigned int n = 0; n < detNum; n++)
			allItems.insert(n);

		for (unsigned int i = 0; i < trkNum; ++i)
			matchedItems.insert(assignment[i]);

		// calculate the difference between allItems and matchedItems, return to unmatchedDetections
		std::set_difference(allItems.begin(), allItems.end(),
			matchedItems.begin(), matchedItems.end(),
			insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
	}
	else if (detNum < trkNum) // there are unmatched trajectory/predictions
	{
		for (unsigned int i = 0; i < trkNum; ++i)
			if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
				unmatchedTrajectories.insert(i);
	}
	else
		;

	// filter out matched with low IOU
	// output matchedPairs
	for (unsigned int i = 0; i < trkNum; ++i)
	{
		if (assignment[i] == -1) // pass over invalid values
			continue;
		if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
		{
			unmatchedTrajectories.insert(i);
			unmatchedDetections.insert(assignment[i]);
		}
		else
			matchedPairs.push_back(cv::Point(i, assignment[i]));
	}
	matched_result.matchedPairs = matchedPairs;
	matched_result.unmatchedDet = unmatchedDetections;
	matched_result.unmatchedTracker = unmatchedTrajectories;
	return matched_result;
};


std::vector<TrackingBox> update_trackers(int _, MatchItems M_items) {
	int f_num = detFrameData.size() - 1;
	std::vector<TrackingBox> Sort_result;
	std::vector<cv::Point> matchedPairs = M_items.matchedPairs;
	std::set<int> unmatchedDetections = M_items.unmatchedDet;

	int detIdx, trkIdx;
	for (unsigned int i = 0; i < matchedPairs.size(); i++)
	{
		trkIdx = matchedPairs[i].x;
		detIdx = matchedPairs[i].y;
		trackers[trkIdx].update(detFrameData[f_num][detIdx].box);
	}

	// create and initialize new trackers for unmatched detections
	for (auto umd : unmatchedDetections)
	{
		KalmanTracker tracker = KalmanTracker(detFrameData[f_num][umd].box);
		trackers.push_back(tracker);
	}

	// get trackers' output
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		if (((*it).m_time_since_update < 1) &&
			((*it).m_hit_streak >= min_hits || f_num <= min_hits))
		{
			TrackingBox res;
			res.box = (*it).get_state();
			res.id = (*it).m_id + 1;
			res.frame = f_num;
			Sort_result.push_back(res);
			it++;
		}
		else
			it++;

		// remove dead tracklet
		if (it != trackers.end() && (*it).m_time_since_update > max_age)
			it = trackers.erase(it);
	}

	//std::cout << "SORT time : " << duration.count() << " ms" << std::endl;

	return Sort_result;
};


void update_dataFrame(int f_num, std::vector<vector<float>> bbox) {
	std::vector<TrackingBox> detData;
	for (int i = 0; i < bbox.size(); i++) {
		TrackingBox tb;
		tb.frame = f_num + 1;
		tb.box = Rect_<float>(cv::Point_<float>(bbox[i][0], bbox[i][1]), cv::Point_<float>(bbox[i][2], bbox[i][3]));
		detData.push_back(tb);
	}
	detFrameData.push_back(detData);
}

std::vector<TrackingBox> get_first_frame_result(int __) {
	int f_num = detFrameData.size() - 1;
	std::vector<TrackingBox> first_frame;
	for (unsigned int i = 0; i < detFrameData[f_num].size(); i++) {
		KalmanTracker trk = KalmanTracker(detFrameData[f_num][i].box);
		trackers.push_back(trk);
	}
	// output the first frame detections
	for (unsigned int id = 0; id < detFrameData[f_num].size(); id++) {
		TrackingBox tb = detFrameData[f_num][id];
		tb.id = id;
		first_frame.push_back(tb);
		//std::cout << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height  << std::endl;
	}
	return first_frame;
}


*/