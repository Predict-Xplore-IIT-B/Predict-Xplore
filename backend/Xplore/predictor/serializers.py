from rest_framework import serializers
from .models import Model

# class ModelSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Model
#         fields = '__all__'

class ModelOptionsSerializer(serializers.Serializer):
    models = serializers.ListField(
        child=serializers.CharField(),
        allow_empty=True
    )

class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Model
        fields = [
            'id', 'name', 'description', 'model_file', 'created_by', 'created_at',
            'model_type', 'model_image', 'allowed_xai_models', 'classes', 'allowed_users'
        ]
        read_only_fields = ['id', 'created_at', 'created_by']
