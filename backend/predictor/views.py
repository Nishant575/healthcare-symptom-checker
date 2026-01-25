from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny

from .models import Prediction, Condition
from .serializers import (
    PredictionRequestSerializer,
    PredictionResponseSerializer,
    PredictionHistorySerializer,
    ConditionSerializer,
)
from .ml_service import get_classifier


class PredictView(APIView):
    
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        symptoms_text = serializer.validated_data['symptoms']
        
        # Get prediction from ML model
        classifier = get_classifier()
        result = classifier.predict(symptoms_text)
        
        # Save prediction to database
        prediction = Prediction.objects.create(
            user=request.user,
            symptoms_text=symptoms_text,
            predicted_condition=result['predicted_condition'],
            confidence=result['confidence'],
            top_5_predictions=result['top_5_predictions']
        )
        
        response_serializer = PredictionResponseSerializer(prediction)
        return Response(response_serializer.data, status=status.HTTP_200_OK)


class PredictionHistoryView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = PredictionHistorySerializer

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user)


class ConditionListView(generics.ListAPIView):

    permission_classes = [AllowAny]
    serializer_class = ConditionSerializer
    queryset = Condition.objects.all()
    pagination_class = None


class ConditionNamesView(APIView):

    permission_classes = [AllowAny]

    def get(self, request):
        classifier = get_classifier()
        conditions = classifier.get_all_conditions()
        return Response({
            'count': len(conditions),
            'conditions': conditions
        })