# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from copy import deepcopy


@MODELS.register_module()
class MTLEncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSampel`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head1: ConfigType,
                 decode_head2: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.decode_head1, self.align_corners1, self.num_classes1, self.out_channels1 = self._init_decode_head(decode_head1)
        self.decode_head2, self.align_corners2, self.num_classes2, self.out_channels2 = self._init_decode_head(decode_head2)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _init_decode_head(self, decode_head: ConfigType):
        """Initialize ``decode_head``"""
        head = MODELS.build(decode_head)
        corners = head.align_corners
        num_classes = head.num_classes
        out_channels = head.out_channels
        return head, corners, num_classes, out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor, which_task) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if which_task == '1' or which_task == '2':
            if self.with_neck:
                x = self.neck(x, which_task)
            return x
        elif which_task == '12':
            if self.with_neck:
                (x1, x2), (loss1, loss2) = self.neck(x, which_task)
            return ((x1,), (loss1,)), ((x2,), (loss2,))

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None, which_task='12')
        return seg_logit

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict], which_task) -> List[Tensor]:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        if which_task == '1':
            x, loss = self.extract_feat(inputs, which_task)
            seg_logits = self.decode_head1.predict(x, batch_img_metas, self.test_cfg)
            return seg_logits
        elif which_task == '2':
            x, loss = self.extract_feat(inputs, which_task)
            seg_logits = self.decode_head2.predict(x, batch_img_metas, self.test_cfg)
            return seg_logits
        elif which_task == '12':
            (x1, loss1), (x2, loss2) = self.extract_feat(inputs, which_task)
            seg_logits1 = self.decode_head1.predict(x1, batch_img_metas, self.test_cfg)
            seg_logits2 = self.decode_head2.predict(x2, batch_img_metas, self.test_cfg)
            return seg_logits1, seg_logits2
        else:
            raise ValueError('Not implemented!')

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList, which_task) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        if which_task == '1':
            loss_decode = self.decode_head1.loss(inputs, data_samples, self.train_cfg)
        elif which_task == '2':
            loss_decode = self.decode_head2.loss(inputs, data_samples, self.train_cfg)
        else:
            raise ValueError('wrong task!')

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        which_task = data_samples[0].which_task
        x, loss_moe = self.extract_feat(inputs, which_task)
        losses.update({'neck.loss': loss_moe[0]})

        loss_decode = self._decode_head_forward_train(x, data_samples, which_task)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        which_task = data_samples[0].which_task
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        if which_task == '1':
            seg_logits = self.inference(inputs, batch_img_metas, which_task)
            post = self.postprocess_result(seg_logits, data_samples, self.align_corners1)
            return post
        elif which_task == '2':
            seg_logits = self.inference(inputs, batch_img_metas, which_task)
            post = self.postprocess_result(seg_logits, data_samples, self.align_corners2)
            return post
        elif which_task == '12':
            seg_logits1, seg_logits2 = self.inference(inputs, batch_img_metas, which_task)
            data_samples1 = deepcopy(data_samples)
            data_samples2 = deepcopy(data_samples)
            post1 = self.postprocess_result(seg_logits1, data_samples1, self.align_corners1)
            post2 = self.postprocess_result(seg_logits2, data_samples2, self.align_corners2)
            return post1, post2
        else:
            raise ValueError('not implemented!')

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        which_task = data_samples[0].which_task
        x, loss = self.extract_feat(inputs, which_task)
        if which_task == '1':
            out = self.decode_head1.forward(x)
        elif which_task == '2':
            out = self.decode_head2.forward(x)
        else:
            raise ValueError('Not implemented!')
        return out

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict], which_task) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        if which_task == '1':
            num_classes = self.num_classes1
            preds = inputs.new_zeros((batch_size, num_classes, h_img, w_img))
        elif which_task == '2':
            num_classes = self.num_classes2
            preds = inputs.new_zeros((batch_size, num_classes, h_img, w_img))
        elif which_task == '12':
            num_classes1 = self.num_classes1
            num_classes2 = self.num_classes2
            preds1 = inputs.new_zeros((batch_size, num_classes1, h_img, w_img))
            preds2 = inputs.new_zeros((batch_size, num_classes2, h_img, w_img))
        else:
            raise ValueError('not implemented!')

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                if which_task == '1' or which_task == '2':
                    crop_seg_logit = self.encode_decode(crop_img, batch_img_metas, which_task)
                    preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                elif which_task == '12':
                    crop_seg_logit1, crop_seg_logit2 = self.encode_decode(crop_img, batch_img_metas, which_task)
                    preds1 += F.pad(crop_seg_logit1, (int(x1), int(preds1.shape[3] - x2), int(y1), int(preds1.shape[2] - y2)))
                    preds2 += F.pad(crop_seg_logit2, (int(x1), int(preds2.shape[3] - x2), int(y1), int(preds2.shape[2] - y2)))
                else:
                    raise ValueError('not implemented!')
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if which_task == '1' or which_task == '2':
            seg_logits = preds / count_mat
            return seg_logits
        elif which_task == '12':
            seg_logits1 = preds1 / count_mat
            seg_logits2 = preds2 / count_mat
            return seg_logits1, seg_logits2
        else:
            raise ValueError('not implemented!')

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict], which_task) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            if which_task == '1' or which_task == '2':
                seg_logit = self.slide_inference(inputs, batch_img_metas, which_task)
                return seg_logit
            elif which_task == '12':
                seg_logit1, seg_logit2 = self.slide_inference(inputs, batch_img_metas, which_task)
                return seg_logit1, seg_logit2
            else:
                raise ValueError('not implemented!')
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)
            return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
