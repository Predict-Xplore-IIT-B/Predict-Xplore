from django.contrib import admin
from .models import Model, Pipeline, TestCase, Report

admin.site.register(Model)
admin.site.register(Pipeline)
admin.site.register(TestCase)
admin.site.register(Report)


# Customised models for trial purposes
# Custom admin class for Model
# @admin.register(Model)
# class ModelAdmin(admin.ModelAdmin):
#     list_display = ('name', 'created_by', 'created_at', 'model_type')  # Fields to display in admin list view
#     search_fields = ('name', 'model_type')  # Enable search by these fields
#     list_filter = ('model_type', 'created_by')  # Filters to narrow down results
#     ordering = ('-created_at',)  # Order by creation date (most recent first)


# # Custom admin class for Pipeline
# @admin.register(Pipeline)
# class PipelineAdmin(admin.ModelAdmin):
#     list_display = ('name', 'created_by', 'is_active', 'created_at')  # Fields to display in admin list view
#     search_fields = ('name',)  # Enable search by name
#     list_filter = ('is_active', 'created_by')  # Filters for activity status and creator
#     ordering = ('-created_at',)  # Order by creation date (most recent first)


# custom admin for TestCase and Report
# @admin.register(TestCase)
# class TestCaseAdmin(admin.ModelAdmin):
#     list_display = ("id", "pipeline", "model", "created_by", "status", "created_at")
#     search_fields = ("pipeline__name", "model__name", "status")
#     list_filter = ("status", "created_at", "pipeline", "model")
#     ordering = ("-created_at")


# @admin.register(Report)
# class ReportAdmin(admin.ModelAdmin):
#     list_display = ("id", "test_case", "created_at")
#     search_fields = ("test_case__id",)
#     ordering = ("-created_at")
