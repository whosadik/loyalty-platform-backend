from rest_framework.throttling import UserRateThrottle


class RecsRateThrottle(UserRateThrottle):
    scope = "recs"


class NextOfferRateThrottle(UserRateThrottle):
    scope = "next_offer"


class CheckoutPreviewRateThrottle(UserRateThrottle):
    scope = "checkout_preview"
