#include "Carla/Sensor/VoxelDetectionSensor.h"
#include "Carla.h"
#include "Carla/rpc/ActorId.h"
#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Carla/Util/BoundingBoxCalculator.h"
#include "Carla/Vehicle/CarlaWheeledVehicle.h"
#include "Kismet/KismetMathLibrary.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Traffic/RoutePlanner.h"
#include "Runtime/Core/Public/Async/ParallelFor.h"

TMap<int32, FLinearColor> AVoxelDetectionSensor::ColorMap = AVoxelDetectionSensor::CreateColorMap();

AVoxelDetectionSensor::AVoxelDetectionSensor(const FObjectInitializer &ObjectInitializer)
  : Super(ObjectInitializer)
{
	PrimaryActorTick.bCanEverTick = true;
}

FActorDefinition AVoxelDetectionSensor::GetSensorDefinition()
{
	auto Definition = UActorBlueprintFunctionLibrary::MakeGenericSensorDefinition(
		TEXT("other"),
		TEXT("voxel_detection"));

	FActorVariation DetectedLen;
	DetectedLen.Id = TEXT("detected_len");
	DetectedLen.Type = EActorAttributeType::Float;
	DetectedLen.RecommendedValues = { TEXT("50.0") };
	DetectedLen.bRestrictToRecommended = false;

	FActorVariation Top;
	Top.Id = TEXT("top_boundary");
	Top.Type = EActorAttributeType::Float;
	Top.RecommendedValues = { TEXT("20.0") };
	Top.bRestrictToRecommended = false;
	
	FActorVariation Bottom;
	Bottom.Id = TEXT("bottom_boundary");
	Bottom.Type = EActorAttributeType::Float;
	Bottom.RecommendedValues = { TEXT("-5.0") };
	Bottom.bRestrictToRecommended = false;

	FActorVariation BoxSize;
	BoxSize.Id = TEXT("box_size");
	BoxSize.Type = EActorAttributeType::Float;
	BoxSize.RecommendedValues = { TEXT("1.0f") };
	BoxSize.bRestrictToRecommended = false;

	FActorVariation SelfIgnore;
	SelfIgnore.Id = TEXT("self_ignore");
	SelfIgnore.Type = EActorAttributeType::Int;
	SelfIgnore.RecommendedValues = { TEXT("1") };
	SelfIgnore.bRestrictToRecommended = false;

	FActorVariation DrawDebug;
	DrawDebug.Id = TEXT("draw_debug");
	DrawDebug.Type = EActorAttributeType::Int;
	DrawDebug.RecommendedValues = { TEXT("1") };
	DrawDebug.bRestrictToRecommended = false;

	Definition.Variations.Append({ DetectedLen, Top, Bottom, BoxSize, SelfIgnore, DrawDebug });

	return Definition;
}

void AVoxelDetectionSensor::Set(const FActorDescription &Description)
{
	Super::Set(Description);

	constexpr float M_TO_CM = 100.0f;
	
	this->DetectedLen = M_TO_CM*UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
		"detected_len",
		Description.Variations,
		50.0f);
	this->Top = this->GetOwner()->GetActorLocation().Z + M_TO_CM*UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
		"top_boundary",
		Description.Variations,
		10.0f);
	this->Bottom = this->GetOwner()->GetActorLocation().Z +  M_TO_CM*UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
		"bottom_boundary",
		Description.Variations,
		-1.0f);

	this->BoxSize = M_TO_CM*UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
		"box_size",
		Description.Variations,
		0.1f);

	this->SelfIgnore = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt(
		"self_ignore",
		Description.Variations,
		1);

	this->DrawDebug = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt(
		"draw_debug",
		Description.Variations,
		0);
}

void AVoxelDetectionSensor::SetOwner(AActor *NewOwner)
{
	Super::SetOwner(NewOwner);
	
}

void AVoxelDetectionSensor::PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaTime)
{
	TArray<AActor*> ActorsToIgnore;
	ActorsToIgnore.Add(this);
	if (SelfIgnore > 0)
	{
		ActorsToIgnore.Add(this->GetOwner());
	}
	auto ac = this->GetOwner();
	Array3D<int32> SemanticVoxels = Array3D<int32>(int(2*this->DetectedLen/this->BoxSize),int(2*this->DetectedLen/this->BoxSize),int((this->Top-this->Bottom)/this->BoxSize), -1);
	Array3D<bool> visited = Array3D<bool>(int(2*this->DetectedLen/this->BoxSize),int(2*this->DetectedLen/this->BoxSize),int((this->Top-this->Bottom)/this->BoxSize), false);
	const FVector Size = FVector{this->DetectedLen, this->DetectedLen, this->Top-this->Bottom};
	const FVector Start = FVector{this->GetOwner()->GetActorLocation().X,this->GetOwner()->GetActorLocation().Y, this->Top};
	const FVector End = FVector{this->GetOwner()->GetActorLocation().X,this->GetOwner()->GetActorLocation().Y, this->Bottom};
	TArray<FHitResult> Hits;
	UKismetSystemLibrary::BoxTraceMultiByProfile(
		GetWorld(),
		Start,
		End,
		Size,
		this->GetOwner()->GetActorRotation(),
		FName("OverlapAll"),
		true,
		ActorsToIgnore,
		EDrawDebugTrace::None,
		Hits,
		true,
		FLinearColor::Red,
		FLinearColor::Green);
	FCriticalSection Mutex;
	TArray<FVector> HitPoints;
    for (auto Hit : Hits)
    {
    	auto CurrentBoxLocation = this->FindNearestBoxLocation(Hit.ImpactPoint);
    	int hitX = (CurrentBoxLocation.X+DetectedLen)/BoxSize - (fmod(CurrentBoxLocation.X+DetectedLen, BoxSize)>0?0:1);
    	int hitY = (CurrentBoxLocation.Y+DetectedLen)/BoxSize - (fmod(CurrentBoxLocation.Y+DetectedLen, BoxSize)>0?0:1);
    	int hitZ = (CurrentBoxLocation.Z-this->Bottom)/BoxSize - (fmod(CurrentBoxLocation.Z-this->Bottom, BoxSize)>0?0:1);
    	if (hitX < 0 || hitX >= SemanticVoxels.getLength() ||
			hitY < 0 || hitY >= SemanticVoxels.getWidth() ||
			hitZ < 0 || hitZ >= SemanticVoxels.getHeight())
		{
			continue;;
		}
    	HitPoints.Emplace(CurrentBoxLocation);
    	visited[hitX][hitY][hitZ] = true;
    }
	this->VoxelDetection(HitPoints, ActorsToIgnore, SemanticVoxels, visited, Mutex);
	UE_LOG(LogTemp, Warning, TEXT("Tick"));
	TArray<int32> DetectedVoxels;
	
	for (int x=0; x<SemanticVoxels.getLength(); ++x)
	{
		for (int y=0; y<SemanticVoxels.getWidth(); ++y)
		{
			for (int z=0; z<SemanticVoxels.getHeight(); ++z)
			{
				if (DrawDebug > 0 && SemanticVoxels[x][y][z] != -1)
				{
					FVector RelatePos = FVector{-DetectedLen+(2*x+1)*BoxSize/2, -DetectedLen+(2*y+1)*BoxSize/2, Bottom+(2*z+1)*BoxSize/2};
					FVector Pos = UKismetMathLibrary::TransformLocation(this->GetOwner()->GetTransform(), RelatePos);
					FLinearColor thisColor = FLinearColor::Green;
					if (ColorMap.Contains(SemanticVoxels[x][y][z]))
					{
						thisColor = ColorMap[SemanticVoxels[x][y][z]];
					}
					thisColor = FLinearColor::Green;
					UKismetSystemLibrary::DrawDebugBox(GetWorld(), Pos, FVector{this->BoxSize/2, this->BoxSize/2, this->BoxSize/2},thisColor, this->GetOwner()->GetActorRotation(), 0.2); // Debug
				}
				DetectedVoxels.Add(SemanticVoxels[x][y][z]);/**/
			}
		}
	}
	auto DataStream = GetDataStream(*this);
	DataStream.Send(*this, GetEpisode(), DetectedVoxels);
}

void AVoxelDetectionSensor::VoxelDetection(TArray<FVector> BoxToDetected, TArray<AActor*>& IgnoreActors, Array3D<int32>& SemanticVoxels, Array3D<bool>& visited, FCriticalSection& Mutex)
{
	int count = 0;
	ParallelFor(BoxToDetected.Num(),[&](int i)
	{
		TArray<FVector> ToDetected;
		ToDetected.Emplace(BoxToDetected[i]);
		while (ToDetected.Num() != 0)
		{
			TArray<FVector> NextBoxToDetect;
			auto num = ToDetected.Num();
			ParallelFor(num,[&](int i)
			// for (int i=0; i < num; i++)
			{
				auto RelateBoxLocation = ToDetected[i];
				int x = (RelateBoxLocation.X+DetectedLen)/BoxSize - (fmod(RelateBoxLocation.X+DetectedLen, BoxSize)>0?0:1);
				int y = (RelateBoxLocation.Y+DetectedLen)/BoxSize - (fmod(RelateBoxLocation.Y+DetectedLen, BoxSize)>0?0:1);
				int z = (RelateBoxLocation.Z-this->Bottom)/BoxSize - (fmod(RelateBoxLocation.Z-this->Bottom, BoxSize)>0?0:1);
				FHitResult OutHits(ForceInit);
				const FVector WorldBoxLocation = UKismetMathLibrary::TransformLocation(this->GetOwner()->GetTransform(), RelateBoxLocation);
				const FVector Start = FVector{WorldBoxLocation.X, WorldBoxLocation.Y, WorldBoxLocation.Z+this->BoxSize/2};
				const FVector End = FVector{WorldBoxLocation.X, WorldBoxLocation.Y, WorldBoxLocation.Z-this->BoxSize/2};
				const FVector Size = FVector{this->BoxSize/2, this->BoxSize/2, this->BoxSize/2};
				UKismetSystemLibrary::BoxTraceSingleByProfile(
					GetWorld(),
					Start,
					End,
					Size,
					this->GetOwner()->GetActorRotation(),
					FName("Vehicle"),
					true,
					IgnoreActors,
					EDrawDebugTrace::None,
					OutHits,
					true);
				if (OutHits.bBlockingHit)
				{
					const auto& CurrentEpisode = GetEpisode();
					SemanticVoxels[x][y][z] = OutHits.Component->CustomDepthStencilValue;
				}else
				{
					return;
				}
				
				Mutex.Lock();
				// x axis expand
				if (x-1 >= 0 && visited[x-1][y][z] == false)
				{
					NextBoxToDetect.Add(FVector{RelateBoxLocation.X-this->BoxSize, RelateBoxLocation.Y, RelateBoxLocation.Z});
					visited[x-1][y][z] = true;
				}
				if (x+1 <= SemanticVoxels.getLength()-1 && visited[x+1][y][z] == false)
				{
					NextBoxToDetect.Add(FVector{RelateBoxLocation.X+this->BoxSize, RelateBoxLocation.Y, RelateBoxLocation.Z});
					visited[x+1][y][z] = true;
				}
				// y axis expand
				if (y-1 >= 0 && visited[x][y-1][z] == false)
				{
					NextBoxToDetect.Add(FVector{RelateBoxLocation.X, RelateBoxLocation.Y-this->BoxSize, RelateBoxLocation.Z});
					visited[x][y-1][z] = true;
				}
				if (y+1 <= SemanticVoxels.getWidth()-1 && visited[x][y+1][z] == false)
				{
					NextBoxToDetect.Add(FVector{RelateBoxLocation.X, RelateBoxLocation.Y+this->BoxSize, RelateBoxLocation.Z});
					visited[x][y+1][z] = true;
				}
				// z axis expand
				if (z-1 >= 0 && visited[x][y][z-1] == false)
				{
					NextBoxToDetect.Add(FVector{RelateBoxLocation.X, RelateBoxLocation.Y, RelateBoxLocation.Z-this->BoxSize});
					visited[x][y][z-1] = true;
				}
				if (z+1 <= SemanticVoxels.getHeight()-1 && visited[x][y][z+1] ==false)
				{
					NextBoxToDetect.Add(FVector{RelateBoxLocation.X, RelateBoxLocation.Y, RelateBoxLocation.Z+this->BoxSize});
					visited[x][y][z+1] = true;
				}
				Mutex.Unlock();
			});
			ToDetected.Empty();
			ToDetected.Append(NextBoxToDetect);
		}
	});
}

const FVector AVoxelDetectionSensor::FindNearestBoxLocation(FVector ImpactPoint)
{
	const FVector relateLocation = UKismetMathLibrary::InverseTransformLocation(this->GetOwner()->GetTransform(), ImpactPoint);
	int x,y,z;
	if (relateLocation.X == -DetectedLen)
	{
		x = 0;
	}
	x = (relateLocation.X+DetectedLen)/BoxSize - (fmod(relateLocation.X+DetectedLen, BoxSize)>0?0:1);
	if (relateLocation.Y == -DetectedLen)
	{
		y = 0;
	}
	y = (relateLocation.Y+DetectedLen)/BoxSize - (fmod(relateLocation.Y+DetectedLen, BoxSize)>0?0:1);
	if (relateLocation.Z == this->Bottom)
	{
		z = 0;
	}
	z = (relateLocation.Z-this->Bottom)/BoxSize - (fmod(relateLocation.Z-this->Bottom, BoxSize)>0?0:1);
	return FVector{-DetectedLen+(2*x+1)*BoxSize/2, -DetectedLen+(2*y+1)*BoxSize/2, Bottom+(2*z+1)*BoxSize/2};
}

