#include "Carla/Sensor/VoxelDetectionSensor.h"
#include "Carla.h"

#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Carla/Util/BoundingBoxCalculator.h"
#include "Carla/Vehicle/CarlaWheeledVehicle.h"

AVoxelDetectionSensor::AVoxelDetectionSensor(const FObjectInitializer &ObjectInitializer)
  : Super(ObjectInitializer)
{
	// Box1 = CreateDefaultSubobject<UBoxComponent>(TEXT("BoxOverlap"));
	// Box1->SetupAttachment(RootComponent);
	// // Box1->SetHiddenInGame(true); // Disable for debugging.
	// Box1->SetCollisionProfileName(FName("OverlapAll"));

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

	Definition.Variations.Append({ DetectedLen, Top, Bottom, BoxSize });

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
	this->Top = M_TO_CM*UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
		"top_boundary",
		Description.Variations,
		10.0f);
	this->Bottom = M_TO_CM*UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
		"bottom_boundary",
		Description.Variations,
		-1.0f);

	this->BoxSize = M_TO_CM*UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
		"box_size",
		Description.Variations,
		0.1f);
	
	// for (int i = 0; i < (Front-Back)/BoxSize; ++i)
	// {
	// 	for (int j = 0; j < (Left-Right)/BoxSize; ++j)
	// 	{
	// 		for (int k = 0; k < (Top-Bottom)/BoxSize; ++k)
	// 		{
	// 			auto * Box = NewObject<UBoxComponent>(this,UBoxComponent::StaticClass(),*FString::Printf(TEXT("Box_%d_%d_%d"), i, j ,k));
	// 			// auto * Box = CreateDefaultSubobject<UBoxComponent>(*FString::Printf(TEXT("Box_%d_%d_%d"), i, j ,k));
	// 			Box->SetupAttachment(RootComponent);
	// 			// Box->SetCollisionProfileName(FName("VoxelDetection"));
	// 			Box->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	// 			// UE_LOG(LogTemp, Error, TEXT("asdfsdf"));
	// 			// UE_LOG(LogTemp, Warning, TEXT("%f %f %f"),Back*M_TO_CM+(i+1/2)*BOX_CM,Right*M_TO_CM+(j+1/2)*BOX_CM,Bottom*M_TO_CM+(k+1/2)*BOX_CM);
	// 			Box->SetRelativeLocation(FVector{Back*M_TO_CM+(i+1/2)*BOX_CM, Right*M_TO_CM+(j+1/2)*BOX_CM, Bottom*M_TO_CM+(k+1/2)*BOX_CM});
	// 			Box->SetBoxExtent(FVector{BOX_CM, BOX_CM, BOX_CM});
	// 			Box->SetHiddenInGame(false); // Disable for debugging.
	// 			Box->ShapeColor = FColor::Red;
	// 			// Box->SetCollisionProfileName(FName("OverlapAll"));
	// 			// Box->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	// 			Box->RegisterComponent();
	// 			Boxes.Add(FString::Printf(TEXT("Box_%d_%d_%d"), i, j ,k),Box);
	// 		}
	// 	}
	// }
	// Box1->SetRelativeLocation(FVector{0.0f, 0.0f, 0.0f});
	// Box1->SetBoxExtent(FVector{10000.0f,10000.0f, 10000.0f});
}

void AVoxelDetectionSensor::SetOwner(AActor *NewOwner)
{
	Super::SetOwner(NewOwner);
	
}

void AVoxelDetectionSensor::PrePhysTick(float DeltaSeconds)
{
    Super::PrePhysTick(DeltaSeconds);

    // TArray<AActor*> DetectedActors;
	TArray<FHitResult> hits;
	// TArray<TEnumAsByte<EObjectTypeQuery>> ObjectTypes;
	// ObjectTypes.Add(UEngineTypes::ConvertToObjectType(ECollisionChannel::ECC_VoxelTrace));
	TArray<AActor*> ActorsToIgnore;
	ActorsToIgnore.Add(this);
	ActorsToIgnore.Add(this->GetParentActor());
	Array3D<AActor*> SemanticVoxels = Array3D<AActor*>(int(2*this->DetectedLen/this->BoxSize),int(2*this->DetectedLen/this->BoxSize),int((this->Top-this->Bottom)/this->BoxSize), nullptr);
	Array3D<bool> visited = Array3D<bool>(int(2*this->DetectedLen/this->BoxSize),int(2*this->DetectedLen/this->BoxSize),int((this->Top-this->Bottom)/this->BoxSize), false);
	const FVector Size = FVector{this->DetectedLen, this->DetectedLen, this->Top-this->Bottom};
	const FVector Start = FVector{this->GetActorLocation().X,this->GetActorLocation().Y, this->Top};
	const FVector End = FVector{this->GetActorLocation().X,this->GetActorLocation().Y, this->Bottom};
	TArray<FHitResult> Hits;
	UKismetSystemLibrary::BoxTraceMultiByProfile(
		GetWorld(),
		Start,
		End,
		Size,
		this->GetActorRotation(),
		FName("VoxelDetection"),
		false,
		ActorsToIgnore,
		EDrawDebugTrace::Persistent,
		Hits,
		true);
    for (auto Hit : Hits)
    {
	    auto HitPoint = Hit.ImpactPoint;
    	auto CurrentBoxLocation = this->FindNearestBoxLocation(HitPoint);
    	this->VoxelDetection(CurrentBoxLocation, ActorsToIgnore, visited, SemanticVoxels);
    }
	// this->VoxelDetection(Start,End,Size,ActorsToIgnore,SemanticVoxels);
	UE_LOG(LogTemp, Warning, TEXT("Tick"));
	TArray<AActor*> DetectedActors;
	for (int x=0; x<SemanticVoxels.getLength(); ++x)
	{
		for (int y=0; y<SemanticVoxels.getWidth(); ++y)
		{
			for (int z=0; z<SemanticVoxels.getHeight(); ++z)
			{
				DetectedActors.Add(SemanticVoxels[x][y][z]);/**/
			}
		}
	}
	auto DataStream = GetDataStream(*this);
	DataStream.SerializeAndSend(*this, GetEpisode(), DetectedActors);
}

void AVoxelDetectionSensor::VoxelDetection(const FVector CurrentBoxLocation, TArray<AActor*>& IgnoreActors, Array3D<bool>& visited, Array3D<AActor*>& SemanticVoxels)
{
	int x = (CurrentBoxLocation.X - (this->GetActorLocation().X-this->DetectedLen))/BoxSize;
	int y = (CurrentBoxLocation.Y - (this->GetActorLocation().Y-this->DetectedLen))/BoxSize;
	int z = (CurrentBoxLocation.Z - this->Bottom)/BoxSize;
	if (x < 0 || x >= visited.getLength() ||
		y < 0 || y >= visited.getWidth() ||
		z < 0 || z >= visited.getHeight() ||
		visited[x][y][z])
	{
		return;
	}
	visited[x][y][z] = true;
	TArray<FHitResult> Hits;
	UKismetSystemLibrary::BoxTraceMultiByProfile(
		GetWorld(),
		FVector{CurrentBoxLocation.X, CurrentBoxLocation.Y, CurrentBoxLocation.Z + this->BoxSize/2},
		FVector{CurrentBoxLocation.X, CurrentBoxLocation.Y, CurrentBoxLocation.Z - this->BoxSize/2},
		FVector{this->BoxSize, this->BoxSize, this->BoxSize},
		this->GetActorRotation(),
		FName("VoxelDetection"),
		false,
		IgnoreActors,
		EDrawDebugTrace::Persistent, //Persistent
		Hits,
		true);
	if (Hits.Num() == 0)
	{
		return;
	}
	UE_LOG(LogTemp, Warning, TEXT("Hit at (%d, %d, %d), num: %d"), x, y, z, Hits.Num());
	const auto& CurrentEpisode = GetEpisode();
	for (auto Hit : Hits)
	{
		auto actor = CurrentEpisode.FindCarlaActor(Hit.GetActor());
		if (actor != nullptr)
		{
			SemanticVoxels[x][y][z] = Hit.GetActor();
		}
	}
	// x axis expand
	if (x-1 >= 0 && !visited[x-1][y][z])
	{
		this->VoxelDetection(
			FVector{CurrentBoxLocation.X-this->BoxSize, CurrentBoxLocation.Y, CurrentBoxLocation.Z},
			IgnoreActors,
			visited,
			SemanticVoxels);
	}
	if (x+1 <= SemanticVoxels.getLength()-1 && !visited[x+1][y][z])
	{
		this->VoxelDetection(
			FVector{CurrentBoxLocation.X+this->BoxSize, CurrentBoxLocation.Y, CurrentBoxLocation.Z},
			IgnoreActors,
			visited,
			SemanticVoxels);
	}
	// y axis expand
	if (y-1 >= 0 && !visited[x][y-1][z])
	{
		this->VoxelDetection(
			FVector{CurrentBoxLocation.X, CurrentBoxLocation.Y-this->BoxSize, CurrentBoxLocation.Z},
			IgnoreActors,
			visited,
			SemanticVoxels);
	}
	if (y+1 <= SemanticVoxels.getWidth()-1 && !visited[x][y+1][z])
	{
		this->VoxelDetection(
			FVector{CurrentBoxLocation.X, CurrentBoxLocation.Y+this->BoxSize, CurrentBoxLocation.Z},
			IgnoreActors,
			visited,
			SemanticVoxels);
	}
	// z axis expand
	if (z-1 >= 0 && !visited[x][y][z-1])
	{
		this->VoxelDetection(
			FVector{CurrentBoxLocation.X, CurrentBoxLocation.Y, CurrentBoxLocation.Z-this->BoxSize},
			IgnoreActors,
			visited,
			SemanticVoxels);
	}
	// auto VoxelStream = GetDataStream(*this);
	// VoxelStream.SerializeAndSend(*this, SemanticVoxels);
}

FVector AVoxelDetectionSensor::FindNearestBoxLocation(FVector ImpactPoint)
{
	float x_pos = this->GetActorLocation().X - this->DetectedLen + int((ImpactPoint.X - (this->GetActorLocation().X - this->DetectedLen))/this->BoxSize)*this->BoxSize + this->BoxSize/2;
	float y_pos = this->GetActorLocation().Y - this->DetectedLen + int((ImpactPoint.Y - (this->GetActorLocation().Y - this->DetectedLen))/this->BoxSize)*this->BoxSize + this->BoxSize/2;
	float z_pos = this->Bottom + ((ImpactPoint.Z - this->Bottom)/this->BoxSize)*this->BoxSize + this->BoxSize/2;
	return FVector{x_pos, y_pos, z_pos};
}

