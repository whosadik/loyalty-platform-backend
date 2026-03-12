HOME_HERO_SLIDES = [
    {
        "id": "main-video",
        "eyebrow": "Эксклюзивно",
        "title": "На первый заказ",
        "description": "Промокод, скидки и специальные предложения для новых покупателей.",
        "button_text": "Узнать подробнее",
        "button_to": "/promotions",
    },
    {
        "id": "jpg-video",
        "eyebrow": "Jean Paul Gaultier",
        "title": "Iconic fragrances",
        "description": "Легендарные ароматы и эффектная бренд-зона с динамичным видео.",
        "button_text": "Смотреть бренд",
        "button_to": "/search?q=Jean%20Paul%20Gaultier",
    },
    {
        "id": "clarins",
        "eyebrow": "Clarins",
        "title": "Уход, который работает мягко",
        "description": "Текстуры, комфорт и ежедневные ритуалы для кожи.",
        "button_text": "Выбрать уход",
        "button_to": "/search?q=Clarins",
    },
    {
        "id": "dalba",
        "eyebrow": "d’Alba",
        "title": "Премиальный glow-уход",
        "description": "Минималистичная подборка для сияния и увлажнения.",
        "button_text": "Смотреть продукты",
        "button_to": "/search?q=d%27Alba",
    },
    {
        "id": "darling",
        "eyebrow": "Darling",
        "title": "Summer essentials",
        "description": "Легкий летний акцент в каталоге ухода и body care.",
        "button_text": "Открыть подборку",
        "button_to": "/search?q=Darling",
    },
]


def get_home_hero_payload():
    return {
        "ok": True,
        "slides": HOME_HERO_SLIDES,
    }
