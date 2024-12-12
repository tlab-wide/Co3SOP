

# <center> COP-3D: Cooperative 3D Occupancy Prediction Dataset and Benchmark by Simulated Ground Truth Detection

- [Benchmark Result](#benchmark-result)
- [Acknowledgements](#acknowledgements)


## Benchmark Result 

**<big> 1. Benchmark Result for Task 1 with Voxel Range [25.6, 25.6, 4.8]m**
<div style="overflow-x: auto;">

| Method <div style="width:100px"> | Modality | mIoU | Empty | Buildings | Fences | Other | Pedestrians | Poles | Roadlines | Roads | Sidewalks | Vegetation | Vehicles | Walls | Trafficsigns | Sky | Ground | Bridge | Railtrack | Guardrail | Trafficlight | Static | Dynamic | Water | Terrain | Unlabeled |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| SSCNet  | Lidar   |  13.21 | 93.01 | 1.84 | 0.16 | 0.00 | 0.00 | 3.60 | 0.00 | 0.23 | 19.22 | 41.43 | 71.73 | 0.26 | 0.00 | 0.00 | 37.73 | 0.00 | 0.00 | 8.22 | 0.25 | 3.68 | 0.07 | 0.00 | 26.41 | 9.26 |
| LMSCNet | Lidar | 24.92 | 96.90 | 8.67 | 22.27 | 0.00 | 0.00 | 29.57 | 2.57 | 86.70 | 42.24 | 43.77 | 85.35 | 9.97 | 18.19 | 0.00 | 62.68 | 0.00 | 0.00 | 12.02 | 0.00 | 18.39 | 1.57 | 0.00 | 36.11 | 21.16 |
| OccFormer  | Camera  | 25.76 | 96.67 | 10.45 | 6.77 | 0.00 | 0.00 | 13.26 | 24.21 | 82.16 | 47.10 | 43.71 | 73.29 | 13.85 | 2.90 | 0.00 | 53.76 | 0.51 | 0.00 | 21.33 | 2.28 | 14.37 | 0.40 | 0.00 | 72.87 | 38.28 |
| SurroundOcc  | Camera  |  26.27 | 97.45 | 8.95 | 8.34 | 0.00 | 0.00 | 10.93 | 26.42 | 86.44 | 48.51 | 44.71 | 74.75 | 6.32 | 9.79 | 0.00 | 59.35 | 0.00 | 0.00 | 44.78 | 0.70 | 6.17 | 1.08 | 0.00 | 52.74 | 43.08 |

</div>

**2. Benchmark Result for Task 2 with Voxel Range [51.2, 51.2, 4.8]m**
<div style="overflow-x: auto;">

| Method <div style="width:100px"> | Modality | mIoU | Empty | Buildings | Fences | Other | Pedestrians | Poles | Roadlines | Roads | Sidewalks | Vegetation | Vehicles | Walls | Trafficsigns | Sky | Ground | Bridge | Railtrack | Guardrail | Trafficlight | Static | Dynamic | Water | Terrain | Unlabeled |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| SSCNet  | Lidar   |  9.58 | 91.18 | 0.17 | 1.48 | 0.00 | 0.00 | 0.14 | 0.16 | 25.88 | 9.57 | 30.89 | 48.09 | 0.49 | 0.00 | 0.00 | 0.08 | 0.03 | 0.00 | 12.72 | 0.00 | 0.94 | 3.09 | 0.00 | 2.74 | 2.31 |
| LMSCNet | Lidar | 20.35 | 95.79 | 3.09 | 18.01 | 0.00 | 0.00 | 24.95 | 0.57 | 75.84 | 48.66 | 34.90 | 75.63 | 10.39 | 0.02 | 0.00 | 31.81 | 0.00 | 0.00 | 6.07 | 0.00 | 4.37 | 0.04 | 0.00 | 36.93 | 21.47 |
| OccFormer  | Camera  |  25.41 | 95.04 | 11.93 | 12.57 | 0.35 | 0.00 | 12.62 | 22.10 | 75.30 | 51.41 | 39.77 | 51.26 | 15.53 | 7.68 | 0.00 | 57.79 | 2.95 | 0.00 | 41.41 | 3.75 | 11.61 | 7.10 | 0.00 | 53.91 | 35.83 |
| SurroundOcc  | Camera  |  22.56 | 95.43 | 5.14 | 9.36 | 2.23 | 0.00 | 2.45 | 21.27 | 77.31 | 48.69 | 31.24 | 53.05 | 11.92 | 1.75 | 0.00 | 49.78 | 1.67 | 0.00 | 35.50 | 1.20 | 8.08 | 2.82 | 0.00 | 47.62 | 34.86 |
| COP3D-Base (Ego)|21.71|95.28|3.67|7.25|1.51|0.00|1.08|19.39|75.08|47.41|27.69|47.80|10.50|0.03|0.00|53.90|0.58|0.00|28.33|0.97|2.30|2.52|0.00|61.65|34.14|
| COP3D-Base (1 CV)| Camera | 24.53 | 95.37 | 4.14 | 11.32 | 0.35| 0.00 |5.02| 26.80 |79.21|45.39|31.75|63.25|12.34|0.22|0.00|48.35|0.35|0.00|39.26|0.87|6.62|4.56|0.00|77.43|36.16|

</div>



## Acknowledgements
Many thanks to these excellent projects:
- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
- [CoHFF](https://github.com/rruisong/CoHFF)
- [LMSCNet](https://github.com/astra-vision/LMSCNet)
- [OccFormer](https://github.com/DerrickXuNu/OpenCOOD)