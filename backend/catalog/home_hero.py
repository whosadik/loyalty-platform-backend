from backend.request_language import AppLanguage, normalize_language


HOME_HERO_SLIDES = [
    {
        "id": "main-video",
        "button_to": "/promotions",
        "translations": {
            "ru": {
                "eyebrow": "Эксклюзивно",
                "title": "На первый заказ",
                "description": "Промокоды, скидки и специальные предложения для новых покупателей.",
                "button_text": "Узнать больше",
            },
            "kk": {
                "eyebrow": "Эксклюзив",
                "title": "Алғашқы тапсырысқа",
                "description": "Жаңа сатып алушыларға арналған промокодтар, жеңілдіктер мен арнайы ұсыныстар.",
                "button_text": "Толығырақ білу",
            },
            "en": {
                "eyebrow": "Exclusive",
                "title": "For your first order",
                "description": "Promo codes, discounts, and special offers for new customers.",
                "button_text": "Learn more",
            },
        },
    },
    {
        "id": "jpg-video",
        "button_to": "/search?q=Jean%20Paul%20Gaultier",
        "translations": {
            "ru": {
                "eyebrow": "Jean Paul Gaultier",
                "title": "Легендарные ароматы",
                "description": "Выразительная бренд-зона и культовые ароматы в одном акценте.",
                "button_text": "Смотреть бренд",
            },
            "kk": {
                "eyebrow": "Jean Paul Gaultier",
                "title": "Аңызға айналған хош иістер",
                "description": "Бір екпінде әсерлі бренд-аймақ пен культтік хош иістер.",
                "button_text": "Брендті көру",
            },
            "en": {
                "eyebrow": "Jean Paul Gaultier",
                "title": "Iconic fragrances",
                "description": "A striking brand zone paired with legendary fragrances.",
                "button_text": "View brand",
            },
        },
    },
    {
        "id": "clarins",
        "button_to": "/search?q=Clarins",
        "translations": {
            "ru": {
                "eyebrow": "Clarins",
                "title": "Уход, который работает мягко",
                "description": "Текстуры, комфорт и ежедневные ритуалы для вашей кожи.",
                "button_text": "Выбрать уход",
            },
            "kk": {
                "eyebrow": "Clarins",
                "title": "Нәзік әсер ететін күтім",
                "description": "Теріңізге арналған текстуралар, жайлылық және күнделікті ритуалдар.",
                "button_text": "Күтімді таңдау",
            },
            "en": {
                "eyebrow": "Clarins",
                "title": "Care that works gently",
                "description": "Textures, comfort, and everyday rituals for your skin.",
                "button_text": "Choose skincare",
            },
        },
    },
    {
        "id": "dalba",
        "button_to": "/search?q=d%27Alba",
        "translations": {
            "ru": {
                "eyebrow": "d'Alba",
                "title": "Премиальный glow-уход",
                "description": "Минималистичная подборка для сияния и увлажнения.",
                "button_text": "Смотреть продукты",
            },
            "kk": {
                "eyebrow": "d'Alba",
                "title": "Премиум glow-күтім",
                "description": "Жарқырау мен ылғалдандыруға арналған минималистік іріктеу.",
                "button_text": "Өнімдерді көру",
            },
            "en": {
                "eyebrow": "d'Alba",
                "title": "Premium glow care",
                "description": "A minimalist edit for glow and hydration.",
                "button_text": "View products",
            },
        },
    },
    {
        "id": "darling",
        "button_to": "/search?q=Darling",
        "translations": {
            "ru": {
                "eyebrow": "Darling",
                "title": "Летние essentials",
                "description": "Легкий сезонный акцент для ухода и body care.",
                "button_text": "Открыть подборку",
            },
            "kk": {
                "eyebrow": "Darling",
                "title": "Жазғы essentials",
                "description": "Күтім мен body care үшін жеңіл маусымдық іріктеу.",
                "button_text": "Іріктеуді ашу",
            },
            "en": {
                "eyebrow": "Darling",
                "title": "Summer essentials",
                "description": "A light seasonal edit for skincare and body care.",
                "button_text": "Open the selection",
            },
        },
    },
]


def get_home_hero_payload(language: AppLanguage = "ru"):
    normalized = normalize_language(language)
    return {
        "ok": True,
        "slides": [
            {
                "id": slide["id"],
                "button_to": slide["button_to"],
                **slide["translations"][normalized],
            }
            for slide in HOME_HERO_SLIDES
        ],
    }
