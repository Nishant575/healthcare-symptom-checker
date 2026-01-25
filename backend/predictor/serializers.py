from rest_framework import serializers
from .models import Prediction, Condition


class PredictionRequestSerializer(serializers.Serializer):
    
    symptoms = serializers.CharField(
        required=True,
        min_length=10,
        help_text="Describe your symptoms"
    )


class PredictionResultSerializer(serializers.Serializer):

    condition = serializers.CharField()
    confidence = serializers.FloatField()


class PredictionResponseSerializer(serializers.ModelSerializer):

    top_5_predictions = serializers.JSONField()
    
    class Meta:
        model = Prediction
        fields = [
            'id',
            'symptoms_text',
            'predicted_condition',
            'confidence',
            'top_5_predictions',
            'created_at',
        ]


class PredictionHistorySerializer(serializers.ModelSerializer):

    class Meta:
        model = Prediction
        fields = [
            'id',
            'symptoms_text',
            'predicted_condition',
            'confidence',
            'created_at',
        ]


class ConditionSerializer(serializers.ModelSerializer):

    class Meta:
        model = Condition
        fields = ['condition_id', 'name', 'description']