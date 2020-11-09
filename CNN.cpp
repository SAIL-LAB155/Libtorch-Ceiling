/*
void CNN_predict(cv::Mat CNN_image) {
	auto input_tensor = torch::from_blob(CNN_image.data, { 1, 224, 224, 3 });
	input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
	input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
	input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
	input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);
	input_tensor = input_tensor.to(at::kCUDA);
	torch::Tensor out_tensor = CNN_model.forward({ input_tensor }).toTensor();

	auto results = out_tensor.sort(-1, true);
	auto softmaxs = std::get<0>(results)[0].softmax(0);
	auto indexs = std::get<1>(results)[0];

	int i = 0;
	auto idx = indexs[i].item<int>();
	std::cout << "    Label:  " << CNN_labels[idx] << "    With Probability:  "
		<< softmaxs[i].item<float>() * 100.0f << "%" << std::endl;
}
*/
