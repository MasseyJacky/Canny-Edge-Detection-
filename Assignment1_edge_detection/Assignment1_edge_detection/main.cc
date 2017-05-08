#include <opencv2\opencv.hpp>  //include all opencv header files

using namespace cv;

void verticle_neibour(Mat intensity, Mat output_matrix, int rows, int cols);
void diagonal_neibour_R(Mat intensity, Mat output_matrix, int rows, int cols);
void horizontal_neibour(Mat intensity, Mat output_matrix, int rows, int cols);
void diagonal_neibour_L(Mat intensity, Mat output_matrix, int rows, int cols);
float check_neibours(Mat input_image, float current_pixel, float upper_threshold, int rows, int cols);

int main(int argc, const char** argv) {

	// Load up the JPG file in the same directory as the project file, 
	// with image depth-8bit and no change in its channel
	Mat image = imread("Image 1.JPG", CV_LOAD_IMAGE_UNCHANGED); 

	// Terminate the program if the image is not loaded successfully
	if (image.empty()) {  
		std::cout << "Error, Image can not be loaded." << std::endl; // a error massage will be displayed
		system("Pause"); 
		return -1; // program endded
	}

	// create a window to show the image
	namedWindow("Original Image", CV_WINDOW_NORMAL); // CV_WINDOW_NORMAL allow user to adjust the window size 
	imshow("Original Image", image);

	// Wait for user input and gives response accordingly
	while (1) {
		std::cout << "Press'Space' to execute 'Canny()'; or Press 'Enter' to execute my function......" << std::endl;
		int key = waitKey(0); // key input
		std::cout << key << std::endl; // shows the figure that repersents the responding key
		std::cout << "Wait till the picture displayed......" << std::endl;

		// "escape" button, to quit the project, close the display window
		if (key == 27) { 
			destroyWindow("Original Image");
			return 0;
		}
		
		// "space" button, canny algorithm, edge detection
		else if (key == 32) { 
			
			Mat gray, edge, draw;

			cvtColor(image, gray, CV_BGR2GRAY);

			Canny(gray, edge, 50, 100, 3);

			namedWindow("edge", CV_WINDOW_NORMAL);
			imshow("edge", edge);
		}

		/***********************************************************************************
		*TRIGGER: this option executes when "enter" is pressed
		*FUNCTION: perform canny edge detection without using canny()
		***********************************************************************************/
		else if (key == 13) {

			Mat image1, gray, Gaussian_filter_kernal;

			cvtColor(image, gray, CV_BGR2GRAY);

			/*******************************************************************************
			* Step 1: Apply gaussian filter to filter out any noises
			*******************************************************************************/
			Gaussian_filter_kernal = (Mat_<double>(5, 5) <<
				2, 4, 5, 4, 2,
				4, 9, 12, 9, 4,
				5, 12, 15, 12, 5,
				4, 9, 12, 9, 4,
				2, 4, 5, 4, 2);
			Gaussian_filter_kernal = Gaussian_filter_kernal / 159;
			filter2D(gray, image1, -1, Gaussian_filter_kernal, Point(-1, -1), 0);

			//cv::GaussianBlur(gray, image1, cv::Size(3, 3), 0, 0);

			/*******************************************************************************
			* Step 2: Find out the intensity-gradiant of the image
			*******************************************************************************/
			//	Sobel(image1, image2, CV_32F, 1, 0, 3); // calculate gradiants X
			//	Sobel(image1, image3, CV_32F, 0, 1, 3); // calculate gradiants Y
			Mat image2 = Mat(image1.rows, image1.cols, CV_32F);
			Mat image3 = Mat(image1.rows, image1.cols, CV_32F);

			Mat sx = (cv::Mat_<double>(3, 3) <<
				-1, 0, 1,
				-2, 0, 2,
				-1, 0, 1);
			Mat sobelx;
			flip(sx, sobelx, 1);

			Mat sobely = (cv::Mat_<double>(3, 3) <<
				-1, -2, -1,
				0, 0, 0,
				1, 2, 1);

			filter2D(image1, image2, CV_32F, sobelx, cv::Point(-1, -1), 0);
			filter2D(image1, image3, CV_32F, sobely, cv::Point(-1, -1), 0);

			image2.convertTo(image2, CV_32F);
			image3.convertTo(image3, CV_32F);

			Mat G = Mat(image1.rows, image1.cols, CV_32F);
			for (int i = 0; i < image2.rows; i++) {
				for (int j = 0; j < image2.cols; j++) {

					float valueX = image2.at<float>(i, j);
					float valueY = image3.at<float>(i, j);
					// Calculate the corresponding single direction, done by applying the arctangens function
					float result = sqrt((valueX*valueX) + (valueY*valueY));
					// Store in orientation matrix element
					G.at<float>(i, j) = result;
				}
			}
			imwrite("test.jpg", G);

			Mat theta = Mat(image2.rows, image3.cols, CV_32F);
			for (int i = 0; i < image2.rows; i++) {
				for (int j = 0; j < image2.cols; j++) {

					float valueX = image2.at<float>(i, j);
					float valueY = image3.at<float>(i, j);
					// Calculate the corresponding single direction, done by applying the arctangens function
					float result = fastAtan2(valueX, valueY);
					// Store in orientation matrix element
					theta.at<float>(i, j) = result;
				}
			}

			/*******************************************************************************
			* Step 3: Apply non maximum suppression
			*******************************************************************************/
			Mat image4 = Mat(image2.rows, image3.cols, CV_32F);
			for (int i = 0; i < image2.rows; i++) {
				for (int j = 0; j < image2.cols; j++) {
					float direction = theta.at<float>(i, j);
					//float intensity = G.at<float>(i, j);
					if ((direction < 22.5 && direction > 337.5) || (direction < 202.5 && direction > 157.5)) {
						//direcion = 0 or 180 degree;
						//compare the neibour pixels
						verticle_neibour(G, image4, i, j);						
					}
					if ((direction < 67.5 && direction > 22.5) || (direction < 247.5 && direction > 202.5)) {
						//direction = 45 or 225 degree;
						//compare the neibour pixels
						diagonal_neibour_R(G, image4, i, j);
					}
					if ((direction < 112.5 && direction > 67.5) || (direction < 292.5 && direction > 247.5)) {
						//direction = 90 or 270 degree;
						//compare the neibour pixels
						horizontal_neibour(G, image4, i, j);						
					}
					if ((direction < 157.5 && direction > 112.5) || (direction < 337.5 && direction > 292.5)) {
						//direction = 135 or 315;
						//compare the neibour pixels
						diagonal_neibour_L(G, image4, i, j);
					}
				}
			}

			/*******************************************************************************
			* Step 4: Apply Hysteresis
			*******************************************************************************/
			float upper_threshold = 70;
			float lower_threshold = upper_threshold / 3;

			Mat image5 = Mat(image4.rows, image4.cols, CV_32F);
			for (int i = 0; i < image4.rows-1; i++) {
				for (int j = 0; j < image4.cols; j++) {
					//check current pixel by pixel
					float current_pixel = image4.at<float>(i, j);
					// if below the threhold, reject
					if (current_pixel < lower_threshold) {
						image5.at<float>(i,j) = 0;
					}
					// else if above the threhold, accept
					else if (current_pixel > upper_threshold) {
						image5.at<float>(i, j) = current_pixel;
					}
					else if ((current_pixel < upper_threshold)&&(current_pixel > lower_threshold)){// current_pixel is between the upper and the lower, exam neibours, if neibour above, accept
						float result = check_neibours(image4, current_pixel, upper_threshold, i, j);
						image5.at<float>(i, j) = result;
					}		
				}
			}
			namedWindow("edge", CV_WINDOW_NORMAL);
			imwrite("NMS.jpg", image5);
			imshow("edge", image5);
		}
	}
	return 0;
}

//functions
void verticle_neibour(Mat intensity, Mat output_matrix, int rows, int cols) {
	float center_pixel = intensity.at<float>(rows, cols);
	float up_pixel = 0;
	float down_pixel = 0;

	//keep neibour pixel valid
	if (rows == 0) {
		//up_pixel = 0;
		down_pixel = intensity.at<float>(rows + 1, cols);
	}
	else if (rows == output_matrix.rows) {
		up_pixel = intensity.at<float>(rows - 1, cols);
		//down_pixel = 0;
	}
	else {
		up_pixel = intensity.at<float>(rows - 1, cols);
		down_pixel = intensity.at<float>(rows + 1, cols);
	}

	//filter unwanted pixels
	if ((center_pixel < up_pixel) || (center_pixel < down_pixel)){
		output_matrix.at<float>(rows, cols) = 0;//not edge, suspress pixel 
	}
	else {		
		output_matrix.at<float>(rows, cols) = center_pixel;//edge, leave pixel
	}
}

void diagonal_neibour_R(Mat intensity, Mat output_matrix, int rows, int cols) {
	float center_pixel = intensity.at<float>(rows, cols);
	float up_right_pixel = 0;
	float down_left_pixel = 0;

	//keep neibour pixel valid
	if ((rows == 0)&&(cols == 0)) {
		//up_right_pixel = 0;
		//down_left_pixel = 0;
	}
	else if (rows == 0){
		//up_right_pixel = 0;
		down_left_pixel = intensity.at<float>(rows + 1, cols - 1);
	}
	else if (cols == 0) {
		up_right_pixel = intensity.at<float>(rows - 1, cols + 1);
		//down_left_pixel = 0;
	}
	else if (cols == output_matrix.cols) {
		//up_right_pixel = 0;
		down_left_pixel = intensity.at<float>(rows + 1, cols - 1);
	}
	else if (rows == output_matrix.rows) {
		up_right_pixel = intensity.at<float>(rows - 1, cols + 1);
		//down_left_pixel = 0;
	}
	else if ((cols == output_matrix.cols)&&(rows == output_matrix.rows)) {
		//up_right_pixel = 0;
		//down_left_pixel = 0;
	}
	else {
		up_right_pixel = intensity.at<float>(rows - 1, cols + 1);
		down_left_pixel = intensity.at<float>(rows + 1, cols - 1);
	}

	//filter unwanted pixels
	if ((center_pixel < up_right_pixel) || (center_pixel < down_left_pixel)) {
		output_matrix.at<float>(rows, cols) = 0;//not edge, suspress pixel 
	}
	else {
		output_matrix.at<float>(rows, cols) = center_pixel;//edge, leave pixel
	}
}

void horizontal_neibour(Mat intensity, Mat output_matrix, int rows, int cols) {
	float center_pixel = intensity.at<float>(rows, cols);
	float left_pixel = 0;
	float right_pixel = 0;

	//keep neibour pixel valid
	if (cols == 0) {
		//left_pixel = 0;
		right_pixel = intensity.at<float>(rows, cols + 1);
	}
	else if (cols == output_matrix.cols) {
		left_pixel = intensity.at<float>(rows, cols - 1);
		//right_pixel = 0;
	}
	else {
		left_pixel = intensity.at<float>(rows, cols - 1);
		right_pixel = intensity.at<float>(rows, cols + 1);
	}

	//filter unwnanted pixels
	if ((center_pixel < left_pixel) || (center_pixel < right_pixel)) {
		output_matrix.at<float>(rows, cols) = 0;//not edge, suspress pixel 
	}
	else {
		output_matrix.at<float>(rows, cols) = center_pixel;//edge, leave pixel
	}
}

void diagonal_neibour_L(Mat intensity, Mat output_matrix, int rows, int cols) {
	float center_pixel = intensity.at<float>(rows, cols);
	float up_left_pixel = 0;
	float down_right_pixel = 0;

	//keep neibour pixel valid
	if (rows == 0) {
		//up_left_pixel = 0;
		down_right_pixel = intensity.at<float>(rows + 1, cols + 1);
	}
	else if ((rows == 0)&&(cols == output_matrix.cols)) {
		//up_left_pixel = 0;
		//down_right_pixel = 0;
	}
	else if (cols == 0){
		//up_left_pixel = 0;
		down_right_pixel = intensity.at<float>(rows + 1, cols + 1);
	}
	else if (cols == output_matrix.cols) {
		up_left_pixel = intensity.at<float>(rows - 1, cols - 1);
		//down_right_pixel = 0;
	}
	else if ((cols == 0) && (rows == 0)) {
		//up_right_pixel = 0;
		//down_left_pixel = 0;
	}
	else if (rows == output_matrix.rows) {
		up_left_pixel = intensity.at<float>(rows - 1, cols - 1);
		//down_left_pixel = 0;
	}
	else {
		up_left_pixel = intensity.at<float>(rows - 1, cols - 1);
		down_right_pixel = intensity.at<float>(rows + 1, cols + 1);
	}

	//filter unwanted pixels
	if ((center_pixel < up_left_pixel) || (center_pixel < down_right_pixel)) {
		output_matrix.at<float>(rows, cols) = 0;//not edge, suspress pixel 
	}
	else {
		output_matrix.at<float>(rows, cols) = center_pixel;//edge, leave pixel
	}
}

float check_neibours(Mat input_image, float current_pixel, float upper_threshold, int rows, int cols) {
	//neighbour pixels
	float up_pixel = 0;
	float down_pixel = 0;
	float up_right_pixel = 0;
	float down_left_pixel = 0;
	float left_pixel = 0;
	float right_pixel = 0;
	float up_left_pixel = 0;
	float down_right_pixel = 0;

	//filter invalid neighbourpixels
	if ((rows == 0)&&(cols == 0)) {
		//float up_pixel = input_image.at<float>(rows - 1, cols);
		down_pixel = input_image.at<float>(rows + 1, cols);
		//float up_right_pixel = input_image.at<float>(rows - 1, cols + 1);
		//float down_left_pixel = input_image.at<float>(rows + 1, cols - 1);
		//float left_pixel = input_image.at<float>(rows, cols - 1);
		right_pixel = input_image.at<float>(rows, cols + 1);
		//float up_left_pixel = input_image.at<float>(rows - 1, cols - 1);
		down_right_pixel = input_image.at<float>(rows + 1, cols + 1);
	}
	else if (rows == 0) {
		//float up_pixel = input_image.at<float>(rows - 1, cols);
		down_pixel = input_image.at<float>(rows + 1, cols);
		//float up_right_pixel = input_image.at<float>(rows - 1, cols + 1);
		down_left_pixel = input_image.at<float>(rows + 1, cols - 1);
		left_pixel = input_image.at<float>(rows, cols - 1);
		right_pixel = input_image.at<float>(rows, cols + 1);
		//float up_left_pixel = input_image.at<float>(rows - 1, cols - 1);
		down_right_pixel = input_image.at<float>(rows + 1, cols + 1);
	}
	else if ((rows == 0) && (cols == input_image.cols)){
		//float up_pixel = input_image.at<float>(rows - 1, cols);
		down_pixel = input_image.at<float>(rows + 1, cols);
		//float up_right_pixel = input_image.at<float>(rows - 1, cols + 1);
		down_left_pixel = input_image.at<float>(rows + 1, cols - 1);
		left_pixel = input_image.at<float>(rows, cols - 1);
		//float right_pixel = input_image.at<float>(rows, cols + 1);
		//float up_left_pixel = input_image.at<float>(rows - 1, cols - 1);
		//float down_right_pixel = input_image.at<float>(rows + 1, cols + 1);
	}
	else if (cols == 0) {
		up_pixel = input_image.at<float>(rows - 1, cols);
		down_pixel = input_image.at<float>(rows + 1, cols);
		up_right_pixel = input_image.at<float>(rows - 1, cols + 1);
		//float down_left_pixel = input_image.at<float>(rows + 1, cols - 1);
		//float left_pixel = input_image.at<float>(rows, cols - 1);
		right_pixel = input_image.at<float>(rows, cols + 1);
		//float up_left_pixel = input_image.at<float>(rows - 1, cols - 1);
		down_right_pixel = input_image.at<float>(rows + 1, cols + 1);
	}
	else if (cols == input_image.cols) {
		up_pixel = input_image.at<float>(rows - 1, cols);
		down_pixel = input_image.at<float>(rows + 1, cols);
		//float up_right_pixel = input_image.at<float>(rows - 1, cols + 1);
		down_left_pixel = input_image.at<float>(rows + 1, cols - 1);
		left_pixel = input_image.at<float>(rows, cols - 1);
		//float right_pixel = input_image.at<float>(rows, cols + 1);
		up_left_pixel = input_image.at<float>(rows - 1, cols - 1);
		//float down_right_pixel = input_image.at<float>(rows + 1, cols + 1);
	}
	else if ((rows == input_image.rows) && (cols == 0)) {
		up_pixel = input_image.at<float>(rows - 1, cols);
		//float down_pixel = input_image.at<float>(rows + 1, cols);
		up_right_pixel = input_image.at<float>(rows - 1, cols + 1);
		//float down_left_pixel = input_image.at<float>(rows + 1, cols - 1);
		//float left_pixel = input_image.at<float>(rows, cols - 1);
		right_pixel = input_image.at<float>(rows, cols + 1);
		//float up_left_pixel = input_image.at<float>(rows - 1, cols - 1);
		//float down_right_pixel = input_image.at<float>(rows + 1, cols + 1);
	}
	else if (rows == input_image.rows) {
		up_pixel = input_image.at<float>(rows - 1, cols);
		//float down_pixel = input_image.at<float>(rows + 1, cols);
		up_right_pixel = input_image.at<float>(rows - 1, cols + 1);
		//float down_left_pixel = input_image.at<float>(rows + 1, cols - 1);
		left_pixel = input_image.at<float>(rows, cols - 1);
		right_pixel = input_image.at<float>(rows, cols + 1);
		up_left_pixel = input_image.at<float>(rows - 1, cols - 1);
		//float down_right_pixel = input_image.at<float>(rows + 1, cols + 1);
	}
	else if ((rows == input_image.rows) && (cols == input_image.cols)) {
		up_pixel = input_image.at<float>(rows - 1, cols);
		//float down_pixel = input_image.at<float>(rows + 1, cols);
		//float up_right_pixel = input_image.at<float>(rows - 1, cols + 1);
		//float down_left_pixel = input_image.at<float>(rows + 1, cols - 1);
		left_pixel = input_image.at<float>(rows, cols - 1);
		//float right_pixel = input_image.at<float>(rows, cols + 1);
		up_left_pixel = input_image.at<float>(rows - 1, cols - 1);
		//float down_right_pixel = input_image.at<float>(rows + 1, cols + 1);
	}
	else{
		up_pixel = input_image.at<float>(rows - 1, cols);
		down_pixel = input_image.at<float>(rows + 1, cols);
		up_right_pixel = input_image.at<float>(rows - 1, cols + 1);
		down_left_pixel = input_image.at<float>(rows + 1, cols - 1);
		left_pixel = input_image.at<float>(rows, cols - 1);
		right_pixel = input_image.at<float>(rows, cols + 1);
		up_left_pixel = input_image.at<float>(rows - 1, cols - 1);
		down_right_pixel = input_image.at<float>(rows + 1, cols + 1);
	}

	//Current pixel is passed only when any of the neighbour pixels is above the upper threshold  
	if ((up_pixel > upper_threshold) || (down_pixel > upper_threshold) || (left_pixel > upper_threshold) || (right_pixel > upper_threshold) || (up_left_pixel > upper_threshold) || (up_right_pixel > upper_threshold) || (down_left_pixel > upper_threshold) || (down_right_pixel > upper_threshold)) {
		return current_pixel;
	}
	else {
		return 0;
	}
}