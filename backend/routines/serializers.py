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
