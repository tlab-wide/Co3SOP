
#pragma once
#include "Carla/Sensor/Sensor.h"
#include "Carla/Actor/ActorDefinition.h"
#include "Carla/Actor/ActorDescription.h"
#include "Components/BoxComponent.h"

#include "VoxelDetectionSensor.generated.h"

template <typename T>
class Array3D
{
public:
	Array3D(int _length, int _width, int _height, const T& defaultValue)
		: length(_length)
		, width(_width)
		, height(_height)
	{
		for (int x = 0; x < length; ++x)
		{
			matrix.Add(TArray<TArray<T>>());
			for (int y = 0; y < width; ++y)
			{
				matrix[x].Add(TArray<T>());
 
				for (int z = 0; z < height; ++z)
				{
					matrix[x][y].Add(defaultValue);
				}
			}
		}
	}
 
	inline TArray<TArray<T>>& operator[](int x)
	{
		return matrix[x];
	}
 
	inline const TArray<TArray<T>>& operator[](int x) const
	{
		return matrix[x];
	}
 
	inline const int getLength() const { return length; }
	inline const int getWidth() const { return width; }
	inline const int getHeight() const { return height; }
 
private:
	TArray<TArray<TArray<T>>> matrix;
	const int length;
	const int width;
	const int height;
};

UCLASS()
class CARLA_API AVoxelDetectionSensor : public ASensor
{
	GENERATED_BODY()

public:

	AVoxelDetectionSensor(const FObjectInitializer &ObjectInitializer);

	static FActorDefinition GetSensorDefinition();

	void Set(const FActorDescription &ActorDescription) override;

	void SetOwner(AActor *NewOwner) override;

	void PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaTime) override;

	void VoxelDetection(TArray<FVector> BoxToDetected, TArray<AActor*>& IgnoreActors, Array3D<int32>& SemanticVoxels, Array3D<bool>& visited, FCriticalSection& Mutex);

	const FVector FindNearestBoxLocation(FVector ImpactPoint);
	
	static TMap<int32, FLinearColor> ColorMap;
	
	static TMap<int32, FLinearColor> CreateColorMap()
	{
		TMap<int32, FLinearColor> ColorMap;
		ColorMap.Add(0,FLinearColor::FromSRGBColor(FColor(0, 0, 0, 0)));
		ColorMap.Add(1,FLinearColor::FromSRGBColor(FColor(70, 70, 70, 255)));
		ColorMap.Add(2,FLinearColor::FromSRGBColor(FColor(190, 153, 153, 255)));
		ColorMap.Add(3,FLinearColor::FromSRGBColor(FColor(55, 90, 80, 255)));
		ColorMap.Add(4,FLinearColor::FromSRGBColor(FColor(220, 20, 60, 255)));
		ColorMap.Add(5,FLinearColor::FromSRGBColor(FColor(153, 153, 153, 255)));
		ColorMap.Add(6,FLinearColor::FromSRGBColor(FColor(157, 234, 50, 255)));
		ColorMap.Add(7,FLinearColor::FromSRGBColor(FColor(128, 64, 128, 255)));
		ColorMap.Add(8,FLinearColor::FromSRGBColor(FColor(244, 35, 232, 255)));
		ColorMap.Add(9,FLinearColor::FromSRGBColor(FColor(107, 142, 35, 255)));
		ColorMap.Add(10,FLinearColor::FromSRGBColor(FColor(0, 0, 142, 255)));
		ColorMap.Add(11,FLinearColor::FromSRGBColor(FColor(102, 102, 156, 255)));
		ColorMap.Add(12,FLinearColor::FromSRGBColor(FColor(220, 220, 0, 255)));
		ColorMap.Add(13,FLinearColor::FromSRGBColor(FColor(70, 130, 180, 255)));
		ColorMap.Add(14,FLinearColor::FromSRGBColor(FColor(81, 0, 81, 255)));
		ColorMap.Add(15,FLinearColor::FromSRGBColor(FColor(150, 100, 100, 255)));
		ColorMap.Add(16,FLinearColor::FromSRGBColor(FColor(230, 150, 140, 255)));
		ColorMap.Add(17,FLinearColor::FromSRGBColor(FColor(180, 165, 180, 255)));
		ColorMap.Add(18,FLinearColor::FromSRGBColor(FColor(250, 170, 30, 255)));
		ColorMap.Add(19,FLinearColor::FromSRGBColor(FColor(110, 190, 160, 255)));
		ColorMap.Add(20,FLinearColor::FromSRGBColor(FColor(170, 120, 50, 255)));
		ColorMap.Add(21,FLinearColor::FromSRGBColor(FColor(45, 60, 150, 255)));
		ColorMap.Add(22,FLinearColor::FromSRGBColor(FColor(145, 170, 100, 255)));
		return ColorMap;
	};
private:
	
	
	float BoxSize = 0.1f;

	float Top = 10.0f;

	float Bottom = -1.0f;

	float DetectedLen = 50.0f;
	
	int SelfIgnore = 1;

	int DrawDebug = 0;
};



