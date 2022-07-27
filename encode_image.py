from ncn_model import CodecNcn

import torchvision
import torch
import timeit
import numpy as np
import matplotlib.pyplot as plt

INPUT_IMAGE_PATH = '/PATH/TO/EXAMPLE/IMAGE.png'  # TODO: Change path to desired image
NCN_MODEL_PATH = './models_to_publish/ncn_maskLoss_CS_maskLatentSpaceNonFrozen_resnetCS_lambda_0_02.pth.tar'  # TODO: Change to desired NCN model path
USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False

if USE_CUDA:
	CUDA_DEVICE = 'cuda'
else:
	CUDA_DEVICE = 'cpu'
DECODE = False  # define whether bitstream shall be created and decoded; takes much longer but returns the real bitrate and no estimated values

### Config file and model path for Mask R-CNN
CONFIG_FILE = "/home/fischer/git/detectron2_for_ncn/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
MODEL_PATH = "/home/fischer/git/detectron2_for_ncn/models/Cityscapes/mask_rcnn_R_50_FPN/model_final_af9cf5.pkl"

torch.use_deterministic_algorithms(True)


def load_cityscapes_model(configFile, modelPath, colorFormat='BGR', returnModel=True, batchSize=None, numWorkers=1,
						  useCuda=True):
	import sys
	sys.path.insert(0, '/home/fischer/git/detectron2_for_ncn/')
	from detectron2.engine import DefaultPredictor
	from detectron2.config import get_cfg
	cfg = get_cfg()
	cfg.merge_from_file(configFile)
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model
	cfg.MODEL.WEIGHTS = modelPath
	cfg.SOLVER.BASE_LR = 0.02
	if batchSize:
		cfg.SOLVER.BATCH_SIZE = batchSize
		cfg.SOLVER.IMS_PER_BATCH = batchSize
	cfg.DATALOADER.NUM_WORKERS = numWorkers
	cfg.INPUT.FORMAT = colorFormat
	if useCuda:
		cfg.MODEL.DEVICE = "cuda"
	else:
		cfg.MODEL.DEVICE = "cpu"
	if returnModel:
		predictor = DefaultPredictor(cfg)
		return predictor, cfg
	else:
		return None, cfg


def plotImage(tensor, idx, plt_show=False):
	imgNP = np.transpose(tensor[idx, :, :, :].cpu().detach().numpy(), [1, 2, 0])
	plt.imshow(imgNP)
	if plt_show:
		plt.show()


class Coder(object):
	def __init__(self, model_path=None, useCuda=True):
		self.ch_top = [192, 192, 192]
		self.model_path = model_path
		self.ch_LS = [192]
		self.ch_pass = [192]
		self.useCuda = useCuda

	def buildModel(self):
		maskLatentSpace = 'maskLatentSpace' in self.model_path
		parallelNet = None
		pixelMean = 0
		pixelStd = 0
		if maskLatentSpace:  # get parallel net from Mask R-CNN model
			predictor, cfg = load_cityscapes_model(CONFIG_FILE, MODEL_PATH, colorFormat='RGB', numWorkers=0,
												   useCuda=USE_CUDA)  # predictor might be required for attention based conv2d network
			import torch.nn as nn
			parallelNet = nn.ModuleList()
			parallelNet.append(predictor.model.backbone.bottom_up.stem)
			parallelNet.append(predictor.model.backbone.bottom_up.res2)
			parallelNet.append(predictor.model.backbone.bottom_up.res3)
			parallelNet.append(predictor.model.backbone.bottom_up.res4)
			pixelMean = predictor.model.pixel_mean.flip(0)  # flip from BGR to RGB
			pixelStd = predictor.model.pixel_std.flip(0)

		self.model = CodecNcn(self.ch_top, self.ch_pass, self.ch_LS, useCuda=self.useCuda, parallelBranch=parallelNet,
							  rcnnMean=pixelMean, rcnnStd=pixelStd, requiresSSIM=False)
		if self.useCuda:
			self.model = self.model.cuda()
		print("Load Model from ", self.model_path)
		if self.useCuda:
			self.model.load_state_dict(torch.load(self.model_path))
		else:
			self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
		self.model.eval()

	def encode(self, img, out=None):
		self.model.path = out
		if self.useCuda:
			img = img.cuda()

		torch.cuda.empty_cache()
		self.model.setApprox(True)

		with torch.no_grad():
			ms_ssim_loss, mse, total_rate, rec = self.model(img)

		return ms_ssim_loss, mse, total_rate, rec


if __name__ == '__main__':
	torch.manual_seed(3)

	start = timeit.default_timer()
	coder = Coder(model_path=NCN_MODEL_PATH, useCuda=USE_CUDA)
	coder.buildModel()
	stop = timeit.default_timer()
	print('Loaded model in ', stop - start)

	inputImg = torchvision.io.read_image(INPUT_IMAGE_PATH)
	inputImg = inputImg.to(torch.float32)[None, :, :, :] / 255

	start = timeit.default_timer()
	path = 'out.ncn'
	if DECODE:
		ms_ssim_loss, mse, total_rate, rec = coder.encode(inputImg, out=path)
	else:
		ms_ssim_loss, mse, total_rate, rec = coder.encode(inputImg)

	stop = timeit.default_timer()
	print(ms_ssim_loss)
	print(mse)
	print(total_rate)
	print('Encoding time: ', stop - start)

	## Activate Decoder to see, whether bitstream is decodable
	if DECODE:
		start = timeit.default_timer()
		decoded = coder.model.decode_bitstream(path)
		stop = timeit.default_timer()
		print('Decoding time: ', stop - start)

	torchvision.utils.save_image(rec, 'coded_image.png')

	plotImage(inputImg, 0)
	plt.figure()
	plotImage(rec, 0)
	if DECODE:
		plt.figure()
		plotImage(decoded, 0)
	plt.show()
