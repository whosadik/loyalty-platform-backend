from rest_framework import serializers


class RoutineGenerateRequestSerializer(serializers.Serializer):
    use_owned = serializers.BooleanField(default=True)


class RoutineValidateRequestSerializer(serializers.Serializer):
    am = serializers.ListField(
        child=serializers.DictField(),
        required=True,
        help_text='Example: [{"step":"cleanser","product_id":1}]',
    )
    pm = serializers.ListField(
        child=serializers.DictField(),
        required=True,
        help_text='Example: [{"step":"serum","product_id":5}]',
    )


class SavedRoutineItemSerializer(serializers.Serializer):
    step = serializers.CharField(max_length=64)
    product_id = serializers.IntegerField(required=False, allow_null=True)


class SavedRoutineWriteSerializer(serializers.Serializer):
    am = serializers.ListField(child=SavedRoutineItemSerializer(), required=True)
    pm = serializers.ListField(child=SavedRoutineItemSerializer(), required=True)
    notes = serializers.ListField(
        child=serializers.CharField(allow_blank=True),
        required=False,
        default=list,
    )
