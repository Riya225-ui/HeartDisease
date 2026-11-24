from pydantic import BaseModel, Field


class HeartDiseaseInput(BaseModel):
    """Input schema for heart disease prediction.

    Features from the UCI Heart Disease dataset.
    """

    age: int = Field(..., description="Age in years", ge=1, le=120, examples=[63])
    sex: int = Field(
        ..., description="Sex (1 = male, 0 = female)", ge=0, le=1, examples=[1]
    )
    cp: int = Field(
        ...,
        description="Chest pain type (0-3)",
        ge=0,
        le=3,
        examples=[3],
    )
    trestbps: int = Field(
        ...,
        description="Resting blood pressure (mm Hg)",
        ge=50,
        le=300,
        examples=[145],
    )
    chol: int = Field(
        ...,
        description="Serum cholesterol (mg/dl)",
        ge=100,
        le=600,
        examples=[233],
    )
    fbs: int = Field(
        ...,
        description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        ge=0,
        le=1,
        examples=[1],
    )
    restecg: int = Field(
        ...,
        description="Resting ECG results (0-2)",
        ge=0,
        le=2,
        examples=[0],
    )
    thalach: int = Field(
        ...,
        description="Maximum heart rate achieved",
        ge=50,
        le=250,
        examples=[150],
    )
    exang: int = Field(
        ...,
        description="Exercise-induced angina (1 = yes, 0 = no)",
        ge=0,
        le=1,
        examples=[0],
    )
    oldpeak: float = Field(
        ...,
        description="ST depression induced by exercise relative to rest",
        ge=0.0,
        le=10.0,
        examples=[2.3],
    )
    slope: int = Field(
        ...,
        description="Slope of the peak exercise ST segment (0-2)",
        ge=0,
        le=2,
        examples=[0],
    )
    ca: int = Field(
        ...,
        description="Number of major vessels colored by fluoroscopy (0-4)",
        ge=0,
        le=4,
        examples=[0],
    )
    thal: int = Field(
        ...,
        description="Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)",
        ge=0,
        le=3,
        examples=[1],
    )


class PredictionResponse(BaseModel):
    """Response schema for heart disease prediction."""

    heart_disease: bool = Field(
        ..., description="Whether heart disease is predicted (true/false)"
    )
    confidence: float = Field(
        ..., description="Prediction confidence (probability)", ge=0.0, le=1.0
    )


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")


class InfoResponse(BaseModel):
    """Response schema for model info."""

    model_type: str = Field(..., description="Type of ML model used")
    feature_list: list[str] = Field(..., description="List of input features")
    dataset: str = Field(..., description="Training dataset name")
    description: str = Field(..., description="Brief description of the model")
