
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
class AVoxelDetectionSensor : public ASensor
{
	GENERATED_BODY()

public:

	AVoxelDetectionSensor(const FObjectInitializer &ObjectInitializer);

	static FActorDefinition GetSensorDefinition();

	void Set(const FActorDescription &ActorDescription) override;

	void SetOwner(AActor *NewOwner) override;

	void PrePhysTick(float DeltaSeconds) override;

	void VoxelDetection(const FVector CurrentBoxLocation, TArray<AActor*>& IgnoreActors, Array3D<bool>& visited, Array3D<AActor*>& SemanticVoxels);

	FVector FindNearestBoxLocation(FVector ImpactPoint);

public:

	// UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Voxels")
	// UBoxComponent *Box1 = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Voxels")
	TMap<FString, UBoxComponent*> Boxes;
private:
	float BoxSize = 0.1f;

	float Top = 10.0f;

	float Bottom = -1.0f;

	float DetectedLen = 50.0f;
};

