from django.db import models
from django.conf import settings


class Prediction(models.Model):

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='predictions'
    )
    symptoms_text = models.TextField()
    predicted_condition = models.CharField(max_length=255)
    confidence = models.FloatField()
    top_5_predictions = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'predictions'
        ordering = ['-created_at']
        verbose_name = 'Prediction'
        verbose_name_plural = 'Predictions'

    def __str__(self):
        return f"{self.user.email} - {self.predicted_condition} ({self.confidence:.2%})"


class Condition(models.Model):
    
    condition_id = models.IntegerField(unique=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)

    class Meta:
        db_table = 'conditions'
        ordering = ['name']

    def __str__(self):
        return self.name