from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .models import CustomerProfile
from .serializers import CustomerProfileSerializer


class MeProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        profile, _ = CustomerProfile.objects.get_or_create(user=request.user)
        return Response(CustomerProfileSerializer(profile).data)

    def put(self, request):
        profile, _ = CustomerProfile.objects.get_or_create(user=request.user)
        serializer = CustomerProfileSerializer(profile, data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)
