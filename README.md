# assignment12-yoloV3
Yolov3 assignment 12 submission with Akta start images


# Steps followed 

## 1. Downloaded images and perform annotation

Downloaded Akta start protein purifier images from internet and annotated these with the help of tool.
Annotation generated the coordinates of objects. Two object we tried to identify. 
1. AktaStart instrument 
2. Sampler

Downloaded and annotated 30 image files. 

## 2. Update yolo code configuration for selected images set and classes

Used the repository [School of AI repo](https://github.com/theschoolofai/YoloV3) and downloaded the code
Updated following files 
1. Created yolov3-aktaStart.cfg. Updated details based on README instruction :
  - filters=21 ((4+1+2)*3)
  - classes=2
  Because I am lazy updated following as well
  - burn_in to 100
  - max_batches to 5000
  - steps to 4000,4500
2. Created folder ./data/aktaStart and following files
  - aktaStart.data
  - aktaStart.names
  - aktaStart.txt : 25 images.
  - aktaStartTest.txt : 15 images. 10 from training and 5 images are not seen during training.


3. Moved images and labels from annotation tools under /data/aktaStart folder
4. Downloaded existing model weights for training


## 3.0 Training 

> python train.py --data data/aktaStart/aktaStart.data --batch-size 10 --cache --cfg cfg/yolov3-aktaStart.cfg --epochs 30 --weights weights/yolov3-spp-ultralytics.pt

Executed for 30 epochs. It will generate following artifacts:
  - New models with name best.py and last.pt
  - train_batch0.png
  - result.png, results.txt, results.json
  - results.txt

![Train Output](./train_batch0.png)

## 3.0 Testing 

> python test.py --data data/aktaStart/aktaStart.data --batch-size 10  --cfg cfg/yolov3-aktaStart.cfg --weights weights/best.pt

Using trained model with name best.pt

I kept batch size 12 just for change. It generated the following output

  - test_batch0.png

![Test Output](./test_batch0.png)

## Execution logs

`
> (pytorch) [ec2-user@ip-10-176-45-7 assignment12-yoloV3]$ python train.py --data data/aktaStart/aktaStart.data --batch 10 --cache --cfg cfg/yolov3-aktaStart.cfg --epochs 3 --nosave


Namespace(epochs=3, batch_size=10, accumulate=4, cfg='cfg/yolov3-aktaStart.cfg', data='data/aktaStart/aktaStart.data', multi_scale=False, img_size=[512], rect=False, resume=False, nosave=True, notest=False, evolve=False, bucket='', cache_images=True, weights='weights/yolov3-spp-ultralytics.pt', name='', device='', adam=False, single_cls=False)
Using CUDA device0 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', total_memory=16151MB)

Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
Model Summary: 225 layers, 6.25787e+07 parameters, 6.25787e+07 gradients
Caching labels (30 found, 0 missing, 0 empty, 0 duplicate, for 30 images): 100%|████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 5288.49it/s]
Caching images (0.0GB): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 117.91it/s]
Caching labels (30 found, 0 missing, 0 empty, 0 duplicate, for 30 images): 100%|████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 5256.02it/s]
Caching images (0.0GB): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 97.84it/s]
Image sizes 512 - 512 train, 512 test
Using 8 dataloader workers
Starting training for 3 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0%|                                                                                                                                                           | 0/3 [00:00<?, ?it/s]/home/ec2-user/my/repo/github/assignment12-yoloV3/utils/utils.py:374: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at /opt/conda/conda-bld/pytorch_1706743807255/work/torch/csrc/tensor/python_tensor.cpp:83.)
  lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
/opt/conda/envs/pytorch/lib/python3.10/site-packages/torch/cuda/memory.py:440: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  warnings.warn(
       0/2     7.85G      4.62       136      1.35       142        32       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:07<00:00,  2.64s/it]
/opt/conda/envs/pytorch/lib/python3.10/site-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1706743807255/work/aten/src/ATen/native/TensorShape.cpp:3549.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:05<00:00,  1.72s/it]
                 all        30        59  0.000103         1  0.000181  0.000206

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
       1/2     7.86G      6.51      95.4      1.72       104        39       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.07it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.14s/it]
                 all        30        59  0.000208     0.619  0.000206  0.000416

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
       2/2     7.86G      6.56      37.9      1.75      46.2        33       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.17it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.13s/it]
                 all        30        59         0         0  0.000245         0
Warning: Plotting error for results.txt, skipping file
3 epochs completed in 0.007 hours.


> (pytorch) [ec2-user@ip-10-176-45-7 assignment12-yoloV3]$ python train.py --data data/aktaStart/aktaStart.data --batch 10 --cache --cfg cfg/yolov3-aktaStart.cfg --epochs 30


Namespace(epochs=30, batch_size=10, accumulate=4, cfg='cfg/yolov3-aktaStart.cfg', data='data/aktaStart/aktaStart.data', multi_scale=False, img_size=[512], rect=False, resume=False, nosave=False, notest=False, evolve=False, bucket='', cache_images=True, weights='weights/yolov3-spp-ultralytics.pt', name='', device='', adam=False, single_cls=False)
Using CUDA device0 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', total_memory=16151MB)

Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
Model Summary: 225 layers, 6.25787e+07 parameters, 6.25787e+07 gradients
Caching labels (30 found, 0 missing, 0 empty, 0 duplicate, for 30 images): 100%|████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 4871.06it/s]
Caching images (0.0GB): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 123.97it/s]
Caching labels (30 found, 0 missing, 0 empty, 0 duplicate, for 30 images): 100%|████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 5198.91it/s]
Caching images (0.0GB): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 94.39it/s]
Image sizes 512 - 512 train, 512 test
Using 8 dataloader workers
Starting training for 30 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0%|                                                                                                                                                           | 0/3 [00:00<?, ?it/s]/home/ec2-user/my/repo/github/assignment12-yoloV3/utils/utils.py:374: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at /opt/conda/conda-bld/pytorch_1706743807255/work/torch/csrc/tensor/python_tensor.cpp:83.)
  lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
/opt/conda/envs/pytorch/lib/python3.10/site-packages/torch/cuda/memory.py:440: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  warnings.warn(
      0/29     7.85G      4.62       136      1.35       142        32       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:05<00:00,  1.82s/it]
/opt/conda/envs/pytorch/lib/python3.10/site-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1706743807255/work/aten/src/ATen/native/TensorShape.cpp:3549.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:05<00:00,  1.72s/it]
                 all        30        59  0.000103         1  0.000181  0.000206

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      1/29     7.86G      6.51        95      1.72       103        39       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.10it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.13s/it]
                 all        30        59  0.000227     0.637  0.000218  0.000453

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      2/29     7.86G      6.55      36.7      1.75        45        33       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.14it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.14s/it]
                 all        30        59         0         0  0.000282         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      3/29     7.86G      5.58      11.8      1.55        19        35       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.21it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.14s/it]
                 all        30        59         0         0  0.000282         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      4/29     7.86G      6.79      7.97      1.74      16.5        31       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.14it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.13s/it]
                 all        30        59         0         0  0.000478         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      5/29     7.86G      7.39      5.28      1.86      14.5        30       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.14it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.02s/it]
                 all        30        59         0         0  0.000838         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      6/29     7.86G      6.04      4.35      1.57        12        37       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.10it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.15it/s]
                 all        30        59         0         0   0.00138         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      7/29       11G      5.65      4.23      1.81      11.7        37       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.17it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.22it/s]
                 all        30        59         0         0   0.00138         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      8/29       11G      5.68      4.33      1.76      11.8        34       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.15it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  4.58it/s]
                 all        30        59         0         0   0.00153         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      9/29       11G      5.36      4.23      1.65      11.2        33       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.14it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.35it/s]
                 all        30        59         0         0   0.00186         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     10/29       11G      6.56      4.49       2.1      13.1        32       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.05it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.34it/s]
                 all        30        59         0         0   0.00158         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     11/29       11G         5      5.57      1.71      12.3        43       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.18it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.37it/s]
                 all        30        59         0         0   0.00158         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     12/29       11G       4.1      5.21      1.34      10.6        34       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.16it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.08it/s]
                 all        30        59         0         0  0.000816         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     13/29       11G      5.35      5.33      1.49      12.2        31       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.12it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.64it/s]
                 all        30        59         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     14/29       11G      4.86      5.04       1.5      11.4        36       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.12it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.43it/s]
                 all        30        59         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     15/29       11G         5      5.81      1.56      12.4        32       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.12it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.59it/s]
                 all        30        59         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     16/29       11G      5.23      5.47      1.73      12.4        29       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.12it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.87it/s]
                 all        30        59         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     17/29       11G      3.47      6.05      1.26      10.8        33       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.07it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.75it/s]
                 all        30        59         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     18/29       11G      4.33      5.68      1.39      11.4        34       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.15it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.39it/s]
                 all        30        59         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     19/29       11G      3.33      6.05      1.38      10.8        46       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.17it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.07it/s]
                 all        30        59         0         0         0         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     20/29       11G       4.6       5.7      1.39      11.7        36       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.12it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.17it/s]
                 all        30        59         0         0   0.00464         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     21/29       11G      4.81      5.42      1.35      11.6        28       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.10it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.24it/s]
                 all        30        59         0         0   0.00464         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     22/29       11G      4.52      5.74      1.39      11.6        35       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.10it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.29it/s]
                 all        30        59         0         0   0.00891         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     23/29       11G      3.49      5.77       1.2      10.5        31       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.16it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.06it/s]
                 all        30        59         0         0   0.00891         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     24/29       11G      4.71      6.15      1.42      12.3        38       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.15it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.92it/s]
                 all        30        59         0         0    0.0139         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     25/29       11G      5.42       5.3      1.72      12.4        29       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.13it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.18it/s]
                 all        30        59         0         0    0.0194         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     26/29       11G      5.22      5.24      1.48      11.9        27       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.12it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.04it/s]
                 all        30        59         0         0    0.0299         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     27/29       11G      5.09      4.69      1.49      11.3        26       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.13it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.06it/s]
                 all        30        59         0         0    0.0299         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     28/29       11G      4.49      6.27      1.69      12.4        37       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.08it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  6.01it/s]
                 all        30        59         0         0    0.0622         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     29/29       11G      3.82      5.15       1.4      10.4        30       512: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.13it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.84it/s]
                 all        30        59       0.5    0.0185    0.0912    0.0357
Warning: Plotting error for results.txt, skipping file
30 epochs completed in 0.053 hours.


> (pytorch) [ec2-user@ip-10-176-45-7 assignment12-yoloV3]$ python test.py --data data/aktaStart/aktaStart.data --batch 10  --cfg cfg/yolov3-aktaStart.cfg


Namespace(cfg='cfg/yolov3-aktaStart.cfg', data='data/aktaStart/aktaStart.data', weights='weights/yolov3-spp-ultralytics.pt', batch_size=10, img_size=416, conf_thres=0.001, iou_thres=0.6, save_json=False, task='test', device='', single_cls=False, augment=False)
Using CUDA device0 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', total_memory=16151MB)

WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
Model Summary: 225 layers, 6.25787e+07 parameters, 6.25787e+07 gradients
Fusing layers...
Model Summary: 152 layers, 6.25519e+07 parameters, 6.25519e+07 gradients
Caching labels (30 found, 0 missing, 0 empty, 0 duplicate, for 30 images): 100%|████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 5003.94it/s]
/opt/conda/envs/pytorch/lib/python3.10/site-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1706743807255/work/aten/src/ATen/native/TensorShape.cpp:3549.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.06it/s]
                 all        30        59       0.5    0.0185     0.132    0.0357
/home/ec2-user/my/repo/github/assignment12-yoloV3/test.py:188: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
           AktaStart        30        32         0         0      0.23         0
             Sampler        30        27         1     0.037     0.035    0.0714
Speed: 7.3/22.8/30.2 ms inference/NMS/total per 416x416 image at batch-size 10
/home/ec2-user/my/repo/github/assignment12-yoloV3/test.py:198: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  maps[c] = ap[i]

> (pytorch) [ec2-user@ip-10-176-45-7 assignment12-yoloV3]$ python test.py --data data/aktaStart/aktaStart.data --batch 10  --cfg cfg/yolov3-aktaStart.cfg

Namespace(cfg='cfg/yolov3-aktaStart.cfg', data='data/aktaStart/aktaStart.data', weights='weights/yolov3-spp-ultralytics.pt', batch_size=10, img_size=416, conf_thres=0.001, iou_thres=0.6, save_json=False, task='test', device='', single_cls=False, augment=False)
Using CUDA device0 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', total_memory=16151MB)

WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
Model Summary: 225 layers, 6.25787e+07 parameters, 6.25787e+07 gradients
Fusing layers...
Model Summary: 152 layers, 6.25519e+07 parameters, 6.25519e+07 gradients
Caching labels (30 found, 0 missing, 0 empty, 0 duplicate, for 30 images): 100%|████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 5175.38it/s]
/opt/conda/envs/pytorch/lib/python3.10/site-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1706743807255/work/aten/src/ATen/native/TensorShape.cpp:3549.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.06it/s]
                 all        30        59       0.5    0.0185     0.132    0.0357
/home/ec2-user/my/repo/github/assignment12-yoloV3/test.py:188: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
           AktaStart        30        32         0         0      0.23         0
             Sampler        30        27         1     0.037     0.035    0.0714
Speed: 7.3/22.8/30.1 ms inference/NMS/total per 416x416 image at batch-size 10
/home/ec2-user/my/repo/github/assignment12-yoloV3/test.py:198: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  maps[c] = ap[i]


> (pytorch) [ec2-user@ip-10-176-45-7 assignment12-yoloV3]$ python test.py --data data/aktaStart/aktaStart.data --batch 12  --cfg cfg/yolov3-aktaStart.cfg


Namespace(cfg='cfg/yolov3-aktaStart.cfg', data='data/aktaStart/aktaStart.data', weights='weights/yolov3-spp-ultralytics.pt', batch_size=12, img_size=416, conf_thres=0.001, iou_thres=0.6, save_json=False, task='test', device='', single_cls=False, augment=False)
Using CUDA device0 _CudaDeviceProperties(name='Tesla V100-SXM2-16GB', total_memory=16151MB)

WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
Model Summary: 225 layers, 6.25787e+07 parameters, 6.25787e+07 gradients
Fusing layers...
Model Summary: 152 layers, 6.25519e+07 parameters, 6.25519e+07 gradients
Caching labels (30 found, 0 missing, 0 empty, 0 duplicate, for 30 images): 100%|████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 5202.35it/s]
/opt/conda/envs/pytorch/lib/python3.10/site-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /opt/conda/conda-bld/pytorch_1706743807255/work/aten/src/ATen/native/TensorShape.cpp:3549.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.08s/it]
                 all        30        59       0.5    0.0185     0.141    0.0357
/home/ec2-user/my/repo/github/assignment12-yoloV3/test.py:188: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
           AktaStart        30        32         0         0     0.247         0
             Sampler        30        27         1     0.037     0.035    0.0714
Speed: 9.8/22.4/32.2 ms inference/NMS/total per 416x416 image at batch-size 12
/home/ec2-user/my/repo/github/assignment12-yoloV3/test.py:198: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  maps[c] = ap[i]


> (pytorch) [ec2-user@ip-10-176-45-7 assignment12-yoloV3]$ 

`
