from django.urls import path
from .views import (
    PredictView,
    PredictionHistoryView,
    ConditionListView,
    ConditionNamesView,
)

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('history/', PredictionHistoryView.as_view(), name='prediction_history'),
    path('conditions/', ConditionListView.as_view(), name='condition_list'),
    path('conditions/names/', ConditionNamesView.as_view(), name='condition_names'),
]