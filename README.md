

# <center> COP-3D: Cooperative 3D Occupancy Prediction Dataset and Benchmark by Simulated Ground Truth Detection

- [Benchmark Result](#benchmark-result)
- [Acknowledgements](#acknowledgements)


## Benchmark Result 

**<big> 1. Benchmark Result for Task 1 with Voxel Range [25.6, 25.6, 4.8]m**
<div style="overflow-x: auto;">

| Method <div style="width:100px"> | Modality | mIoU | Empty | Buildings | Fences | Other | Pedestrians | Poles | Roadlines | Roads | Sidewalks | Vegetation | Vehicles | Walls | Trafficsigns | Sky | Ground | Bridge | Railtrack | Guardrail | Trafficlight | Static | Dynamic | Water | Terrain | Unlabeled |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| SSCNet  | Lidar   |  13.21 | 93.01 | 1.84 | 0.16 | 0.00 | 0.00 | 3.60 | 0.00 | 0.23 | 19.22 | 41.43 | 71.73 | 0.26 | 0.00 | 0.00 | 37.73 | 0.00 | 0.00 | 8.22 | 0.25 | 3.68 | 0.07 | 0.00 | 26.41 | 9.26 |
| LMSCNet | Lidar | 18.40 | 96.98 | 0.08 | 0.16 | 0.00 | 0.00 | 0.02 | 0.07 | 87.94 | 42.30 | 12.78 | 76.82 | 0.38 | 0.00 | 0.00 | 58.23 | 0.00 | 0.00 | 2.30 | 0.00 | 0.00 | 0.00 | 0.00 | 48.15 | 15.55 |
| OccFormer  | Camera  | 19.32 | 96.44 | 6.55 | 4.49 | 0.00 | 0.00 | 0.44 | 12.13 | 71.57 | 43.11 | 28.66 | 45.19 | 7.42 | 0.00 | 0.00 | 21.42 | 0.00 | 0.00 | 31.09 | 0.00 | 3.92 | 0.13 | 0.00 | 59.45 | 31.78 |
| SurroundOcc  | Camera  |  0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

</div>

**2. Benchmark Result for Task 2 with Voxel Range [51.2, 51.2, 4.8]m**
<div style="overflow-x: auto;">

| Method <div style="width:100px"> | Modality | mIoU | Empty | Buildings | Fences | Other | Pedestrians | Poles | Roadlines | Roads | Sidewalks | Vegetation | Vehicles | Walls | Trafficsigns | Sky | Ground | Bridge | Railtrack | Guardrail | Trafficlight | Static | Dynamic | Water | Terrain | Unlabeled |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| SSCNet  | Lidar   |  9.58 | 91.18 | 0.17 | 1.48 | 0.00 | 0.00 | 0.14 | 0.16 | 25.88 | 9.57 | 30.89 | 48.09 | 0.49 | 0.00 | 0.00 | 0.08 | 0.03 | 0.00 | 12.72 | 0.00 | 0.94 | 3.09 | 0.00 | 2.74 | 2.31 |
| LMSCNet | Lidar | 13.42 | 94.04 | 0.00 | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | 69.57 | 33.18 | 0.09 | 49.74 | 0.01 | 0.00 | 0.00 | 18.27 | 0.00 | 0.00 | 1.14 | 0.00 | 0.00 | 0.00 | 0.00 | 39.22 | 16.86 |
| OccFormer  | Camera  | 14.21 | 95.24 | 1.19 | 2.46 | 0.00 | 0.00 | 0.00 | 6.29 | 56.39 | 39.73 | 13.75 | 6.80 | 4.41 | 0.00 | 0.00 | 28.57 | 0.00 | 0.00 | 29.13 | 0.00 | 0.00 | 0.00 | 0.00 | 28.12 | 28.85 |
| SurroundOcc  | Camera  |  0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

</div>



## Acknowledgements
Many thanks to these excellent projects:
- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
- [CoHFF](https://github.com/rruisong/CoHFF)
- [LMSCNet](https://github.com/astra-vision/LMSCNet)
- [OccFormer](https://github.com/DerrickXuNu/OpenCOOD)