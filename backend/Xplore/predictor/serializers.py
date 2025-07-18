from rest_framework import serializers
from .models import Model
from django.conf import settings

# class ModelSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Model
#         fields = '__all__'

class ModelOptionsSerializer(serializers.Serializer):
    models = serializers.ListField(
        child=serializers.CharField(),
        allow_empty=True
    )
    xai_algo = serializers.CharField(
        allow_blank=True,
        required=False,
        help_text="Which XAI algorithm to run (e.g. 'gradcam')"
    )

class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Model
        fields = [
            'id', 'name', 'description', 'model_file', 'created_by', 'created_at',
            'model_type', 'model_image', 'allowed_xai_models', 'classes', 'allowed_users'
        ]
        read_only_fields = ['id', 'created_at', 'created_by']


class PredictSerializer(serializers.Serializer):
    username  = serializers.CharField()
    models    = serializers.ListField(child=serializers.CharField())
    xai_algo  = serializers.ChoiceField(
       choices=[key for key, _ in settings.XAI_ALGOS],
       allow_null=True,
       default=None
    )


