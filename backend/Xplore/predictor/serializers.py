from rest_framework import serializers
from .models import Model
from django.conf import settings

# This serializer is used by your PredictView to validate the incoming data
# for an inference request.
class ModelOptionsSerializer(serializers.Serializer):
    # ADD THIS FIELD: It captures the ID from the image upload step.
    test_case_id = serializers.IntegerField()
    
    models = serializers.ListField(
        child=serializers.IntegerField(),  
        allow_empty=True
    )
    xai_algo = serializers.CharField(
        allow_blank=True,
        required=False,
        help_text="Which XAI algorithm to run (e.g. 'gradcam')"
    )
    target_class = serializers.CharField(
        allow_blank=True,
        required=False,
        help_text="The target class for XAI analysis (e.g. 'forest')"
    )

class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Model
        fields = [
            'id', 'name', 'description', 'model_file', 'created_by', 'created_at',
            'model_type', 'model_image', 'allowed_xai_models', 'classes', 'allowed_users'
        ]
        read_only_fields = ['id', 'created_at', 'created_by']


# NOTE: This serializer is likely no longer used by the new PredictView.
# You can consider removing it to avoid confusion.
class PredictSerializer(serializers.Serializer):
    username  = serializers.CharField()
    models    = serializers.ListField(child=serializers.CharField())
    xai_algo  = serializers.ChoiceField(
        choices=[key for key, _ in settings.XAI_ALGOS],
        allow_null=True,
        default=None
    )
