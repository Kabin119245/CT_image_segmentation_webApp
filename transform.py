from monai.transforms import(
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)


#Defining Transformations
test_transforms = Compose(
    [
        LoadImaged(keys=["vol"]),
        EnsureChannelFirstD(keys=["vol"]),
        Spacingd(keys=["vol"], pixdim=(1.5, 1.5, 1.0), mode=("bilinear")),
        Orientationd(keys=["vol"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['vol'], source_key='vol'),
        Resized(keys=["vol"], spatial_size=[128, 128, 64]),
        ToTensord(keys=["vol"]),
    ]
)