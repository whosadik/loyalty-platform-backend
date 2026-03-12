from rest_framework.throttling import UserRateThrottle


class RecsRateThrottle(UserRateThrottle):
    scope = "recs"


class NextOfferRateThrottle(UserRateThrottle):
    scope = "next_offer"


class CheckoutPreviewRateThrottle(UserRateThrottle):
    scope = "checkout_preview"


class AuthLoginRateThrottle(UserRateThrottle):
    scope = "auth_login"


class AuthRegisterRateThrottle(UserRateThrottle):
    scope = "auth_register"


class AuthPasswordResetRequestRateThrottle(UserRateThrottle):
    scope = "auth_password_reset_request"


class AuthVerifyEmailResendRateThrottle(UserRateThrottle):
    scope = "auth_verify_email_resend"
