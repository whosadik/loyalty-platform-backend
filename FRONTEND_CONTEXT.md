# Frontend Context for `loyalty-platform-backend`

## 1. Что это за backend

Платформа лояльности для бьюти-ритейла с персонализацией:

- каталог товаров (`skincare`, `haircare`, `makeup`, `fragrance`);
- профиль пользователя;
- баллы и уровни лояльности;
- персональные офферы;
- checkout с применением оффера и списанием/начислением баллов;
- рекомендации (For You / Because You Bought / Trending);
- roadmap по уходу/категориям;
- admin API для аналитики, аудита и кампаний.

Стек:

- Django 5 + DRF;
- PostgreSQL (`127.0.0.1:5433`, db/user/pass: `loyalty`);
- Redis (`127.0.0.1:6379`);
- OpenAPI via drf-spectacular.

## 2. Базовые правила интеграции

### База API

- Базовый префикс: `/api/`
- Swagger UI: `/api/docs/`
- OpenAPI schema: `/api/schema/`

### Аутентификация

- В `REST_FRAMEWORK` включен только `SessionAuthentication`.
- По умолчанию для API требуется `IsAuthenticated`.
- Для `POST/PATCH/PUT/DELETE` при session auth нужен CSRF token.
- Есть endpoint для DRF login/logout: `/api-auth/`.

### Заголовки

- Middleware всегда возвращает `X-Request-ID`.
- Можно присылать свой `X-Request-ID` для идемпотентности/трассировки событий.

### Формат ошибок

Глобальный формат ошибок (кастомный handler):

```json
{
  "ok": false,
  "code": "validation_error|error|server_error|...",
  "message": "Validation error|...",
  "details": {}
}
```

Важно: часть view возвращает `400` вручную в формате:

```json
{"ok": false, "message": "..."}
```

Фронту нужно поддерживать оба варианта.

### Rate limits

- anon: `60/min`
- authenticated user: `240/min`
- recs scope (`/api/me/recommendations*`, event): `30/min`
- next offer (`/api/me/next-offer`): `20/min`
- checkout preview: `30/min`

### Пагинация

- Глобальной пагинации нет.
- Пагинация есть только у `/api/admin/audit` (`page`, `page_size`, max `200`).

### Формат чисел/дат

- Денежные поля часто приходят строкой (`"24.68"`), не number.
- `Decimal` в JSON почти везде сериализуется как string.
- Даты/время в ISO-формате.

### Трейлинг-слеши

- Router endpoints со слешем: `/api/products/`, `/api/transactions/`, `/api/me/owned-products/`.
- Большинство custom endpoints без слеша: `/api/me/profile`, `/api/checkout`, ...

Нужно использовать exact path.

## 3. Главные сущности (для UI/типов)

- `Product`: id, name, brand, price, category, product_type, attrs/concerns/flags, in_stock, контент карточки.
- `CustomerProfile`: skin_type, goals, avoid_flags, budget, hair/makeup/fragrance profiles.
- `LoyaltyAccount`: tier, points_balance.
- `OfferAssignment`: assigned offer + target + reason + expires_at + redeemed flag.
- `Transaction`: checkout результат + items.
- `OwnedProduct`: что пользователь уже купил/использует.
- `RoadmapPlan` + `RoadmapStep`: персональный roadmap и следующий шаг.
- `RecommendationEvent`: impression/click/add_to_cart/purchase_attributed.
- `OfferEvent`: assigned/exposed/clicked/redeemed/expired/superseded.

## 4. Публичное user API (основной фронт)

## 4.1 Catalog

### `GET /api/products/`

Фильтры query:

- `category`
- `product_type`
- `brand`
- `in_stock=true|false`

Ответ: массив `Product`.

### `GET /api/products/{id}/`

Ответ: `Product`.

### `POST /api/products/`, `PUT/PATCH/DELETE /api/products/{id}/`

Технически доступны любому authenticated user (в проекте нет staff-ограничения на этот viewset).

## 4.2 Профиль пользователя

### `GET /api/me/profile`

Ответ: объект профиля (`CustomerProfile`).

### `PUT /api/me/profile`

Partial update профиля.

Пример body:

```json
{
  "skin_type": "sensitive",
  "goals": ["hydration"],
  "budget": "medium",
  "hair_profile": {},
  "makeup_profile": {},
  "fragrance_profile": {}
}
```

Ответ:

```json
{
  "ok": true,
  "profile": {...},
  "profile_completion_bonus": {
    "ok": true,
    "awarded": true,
    "points_added": 50,
    "completed": true
  }
}
```

### `GET /api/me/favorite-category`

Ответ:

```json
{
  "ok": true,
  "favorite_category": "makeup|null",
  "window_days": 90,
  "profile_complete": true,
  "explain": {
    "window_start": "...",
    "window_end": "...",
    "history_items_considered": 12,
    "picked_by": ["total_qty","line_count","last_at","category"],
    "signals": [...]
  }
}
```

## 4.3 Loyalty

### `GET /api/me/loyalty`

```json
{"tier": "Bronze", "points_balance": 123}
```

### `POST /api/loyalty/redeem-points`

Body:

```json
{"points": 50, "reference": "optional"}
```

Успех:

```json
{"ok": true, "new_balance": 73}
```

Ошибка:

```json
{"ok": false, "message": "Insufficient points"}
```

## 4.4 Recommendations

### `GET /api/me/recommendations`

Query:

- `category` (`skincare|haircare|makeup|fragrance`, optional)
- `product_type` (optional)
- `limit` (1..50, default 10)
- `algo` (`cooc|reranker|auto`, optional)

Ответ:

```json
{
  "query": {
    "category": "makeup",
    "product_type": null,
    "limit": 10,
    "algo_requested": "reranker",
    "algo_used": "reranker|cooc|cold_start:trending|cooc_fallback:*",
    "model_version": "..."
  },
  "context": {...},
  "results": [
    {
      "product": {...},
      "score": 0.91,
      "components": {...},
      "why": [...]
    }
  ]
}
```

### `GET /api/me/recommendations/bundle`

Query:

- `product_id` (required)
- `limit`
- `algo`

Ответ: `query + results[]` аналогично.

### `GET /api/me/recommendations/home`

Query:

- `limit`
- `category`
- `product_type`
- `price_min`
- `price_max`
- `algo`

Ответ:

```json
{
  "ok": true,
  "query": {...},
  "sections": [
    {"key": "for_you", "title": "For you", "results": [...]},
    {"key": "because_you_bought", "title": "...", "base_product_id": 123, "results": [...]},
    {"key": "trending", "title": "Trending", "results": [...]}
  ]
}
```

Важный side-effect: endpoint пишет `impression` events автоматически.

### `POST /api/me/recommendations/event`

Для клиентских событий.

Body:

```json
{
  "action": "click|add_to_cart",
  "product_id": 123,
  "page": "home",
  "section_key": "for_you",
  "context": {}
}
```

Ответ: `{"ok": true}`.

## 4.5 Offers

### `GET /api/me/next-offer`

Возвращает текущий активный assignment или создает новый.

```json
{
  "assignment_id": 4,
  "offer": {"id": 1, "name": "...", "type": "discount|points_multiplier", "value": "10.00"},
  "target": {"scope": "cart|category|product_type|product_id", "...": "..."},
  "reason": {...},
  "expires_at": "..."
}
```

Также пишет `offer_exposed` (идемпотентно по request key).

### `GET /api/me/offers`

Список активных офферов пользователя (до 50), каждый с `assignment_id`, `offer`, `target`, `reason`, `expires_at`.

### `POST /api/offers/preview`

Body:

```json
{
  "assignment_id": 4,
  "items": [{"product": 330, "quantity": 1}]
}
```

Ответ:

```json
{
  "ok": true,
  "assignment_id": 4,
  "offer": {...},
  "target": {...},
  "gross_total": "25.98",
  "eligible_total": "12.99",
  "discount_amount": "1.30",
  "net_total": "24.68",
  "estimated_points_earned": 25,
  "points_multiplier": "1",
  "tier": "Bronze",
  "points_rate": "1.0"
}
```

### `POST /api/offers/click`

Body:

```json
{"assignment_id": 4, "context": {}}
```

Ответ:

```json
{"ok": true, "assignment_id": 4, "clicked_recorded": true}
```

### `POST /api/offers/redeem`

Body:

```json
{"assignment_id": 4, "transaction_id": 321}
```

Важно:

- для `discount` офферов нужно применять через `/api/checkout` (`apply_assignment_id`);
- этот endpoint практичен для `points_multiplier`.

Успех:

```json
{
  "ok": true,
  "earned_points": 42,
  "new_balance": 130,
  "tier": "Bronze",
  "discount_amount": "0.00"
}
```

## 4.6 Checkout

### `POST /api/checkout/preview`

Body:

```json
{
  "channel": "offline|online",
  "items": [{"product": 330, "quantity": 1}],
  "apply_assignment_id": 4,
  "redeem_points": 10
}
```

Ответ:

```json
{
  "ok": true,
  "gross_total": "25.98",
  "discount_amount": "1.30",
  "net_total": "24.68",
  "offer_applied": true,
  "applied_offer": {...},
  "target": {...},
  "eligible_total": "12.99",
  "points_rate": "1.0",
  "estimated_points_earned": 25,
  "points_redeemed": 10,
  "balance_before": 121,
  "balance_after_estimated": 136,
  "tier": "Bronze"
}
```

### `POST /api/checkout`

Body:

```json
{
  "idempotency_key": "checkout-uuid-1",
  "channel": "offline",
  "items": [{"product": 330, "quantity": 1}],
  "apply_assignment_id": 4,
  "redeem_points": 10
}
```

Успех (`201`):

```json
{
  "ok": true,
  "transaction_id": 123,
  "gross_total": "25.98",
  "discount_amount": "1.30",
  "net_total": "24.68",
  "offer_applied": true,
  "offer_assignment_id": 4,
  "target": {...},
  "eligible_total": "12.99",
  "points_redeemed": 10,
  "points_earned": 25,
  "new_balance": 136,
  "tier": "Bronze",
  "next_offer": {...}
}
```

Идемпотентный повтор с тем же `idempotency_key`:

- статус `200`;
- `idempotent_replay: true`;
- тот же `transaction_id` и payload.

## 4.7 Transactions + Owned products

### `GET /api/transactions/`

Список транзакций пользователя (по убыванию `created_at`).

### `GET /api/transactions/{id}/`

Детали одной транзакции.

### `GET /api/me/owned-products/`

Список owned продуктов с вложенным `product`.

### `GET /api/me/owned-products/{id}/`

Детали owned-продукта.

### `PATCH /api/me/owned-products/{id}/`

Частичное обновление (например `is_active`).

### `POST /api/me/owned-products/{id}/activate/`

`{"ok": true, "id": 7, "is_active": true}`

### `POST /api/me/owned-products/{id}/deactivate/`

`{"ok": true, "id": 7, "is_active": false}`

### `POST /api/me/owned-products/`

Запрещен (`405`, create отключен).

## 4.8 Routine

### `POST /api/routine/generate`

Body:

```json
{"use_owned": true}
```

Ответ: runtime-generated routine (`am`, `pm`, `notes` и т.д.).

### `POST /api/routine/validate`

Body:

```json
{
  "am": [{"step": "cleanser", "product_id": 1}],
  "pm": [{"step": "serum", "product_id": 5}]
}
```

Ответ: результат валидации рутины.

## 4.9 Roadmap

### `GET /api/me/roadmap?category=...`

`category`: `skincare|haircare|makeup|fragrance`.

Если category не передан и активного roadmap нет -> `400`.

Ответ:

```json
{
  "id": 7,
  "category": "haircare",
  "is_active": true,
  "version": 1,
  "meta": {...},
  "steps": [
    {
      "id": 13,
      "step_index": 2,
      "product_type": "conditioner",
      "status": "missing|recommended|owned|skipped|completed",
      "recommended_product": {...},
      "suggestions": [18,19,20],
      "score": 0.72,
      "confidence": 0.68,
      "why": [...],
      "cadence": "daily|weekly|optional"
    }
  ],
  "summary": {
    "next_step": {...},
    "missing_steps_count": 3,
    "total_steps": 5
  }
}
```

### `POST /api/me/roadmap/refresh`

Body:

```json
{"category": "haircare"}
```

Пересобирает roadmap по категории.

### `PATCH /api/me/roadmap/steps/{step_id}`

Body:

```json
{"status": "missing|recommended|owned|skipped|completed"}
```

Ответ:

```json
{"ok": true, "step": {...}}
```

### `POST /api/me/roadmap/steps/{step_id}/click`

Ответ: `{"ok": true, "step_id": 42}`.

## 5. Admin API (для admin frontend)

Доступ управляется `user.is_staff` + `StaffProfile` + permission codes.

Роли:

- `admin`: `view_metrics`, `view_audit`, `invalidate_cache`, `manage_campaigns`, `manage_offers`
- `manager`: `view_metrics`, `invalidate_cache`, `manage_campaigns`, `manage_offers`
- `analyst`: `view_metrics`, `view_audit`

## 5.1 Health and metrics

### `GET /api/admin/health`

- Permission: `IsAdminUser` (staff/superuser style check)
- Ответ: db/cache ok + counters + server_time

### `GET /api/admin/metrics`

- Permission: `view_metrics`
- Большой payload: offers, budget, loyalty, routines, segments, retention, recs, campaigns

### `GET /api/admin/overview`

- Permission: `view_metrics`
- Dashboard payload блоками: transactions, points, offers lifecycle, retention, recs

### `GET /api/admin/recs/experiments`

- Permission: `view_metrics`
- Query: `days`, `experiment_id`, `variant`
- Ответ: summary + experiments breakdown (CTR/CR per variant)

## 5.2 Campaign management

### `GET /api/admin/campaigns`

- Permission: `view_metrics`
- Query: `is_active`, `name`, `ordering`

### `POST /api/admin/campaigns`

- Permission: `manage_campaigns`
- Создает campaign budget

### `GET /api/admin/campaigns/{id}`

- Permission: `view_metrics`

### `PATCH /api/admin/campaigns/{id}`

- Permission: `manage_campaigns`
- Поддерживает `reset_weekly_spent=true`

## 5.3 Audit and cache

### `GET /api/admin/audit`

- Permission: `view_audit`
- Фильтры: `action`, `user_id`, `request_id`, `entity_type`, `entity_id`, `path`, `status_code`, `since`, `until`
- Пагинация: `page`, `page_size`

### `GET /api/admin/audit/export.csv`

- Permission: `view_audit`
- Те же фильтры, stream CSV

### `POST /api/admin/cache/invalidate`

- Permission: `invalidate_cache`
- Сбрасывает recs cache keys

## 6. Рекомендуемый фронтовый flow

1. `GET /api/me/profile` + `GET /api/me/loyalty`
2. Home:
   - `GET /api/me/recommendations/home`
   - при клике/ATC: `POST /api/me/recommendations/event`
3. Offers:
   - `GET /api/me/next-offer` (или `GET /api/me/offers`)
   - click telemetry: `POST /api/offers/click`
4. Cart:
   - `POST /api/checkout/preview`
   - `POST /api/checkout` c `idempotency_key`
5. После checkout:
   - обновить loyalty (`GET /api/me/loyalty`)
   - обновить offers (`GET /api/me/next-offer`)
   - обновить roadmap (`GET /api/me/roadmap?category=...`)

## 7. Потенциальные подводные камни

- Микс форматов ошибок (`details` envelope vs `{ok:false,message}`).
- Денежные значения приходят строками.
- Не все endpoints возвращают поле `ok`.
- Mixed slash policy по роутам.
- `POST /api/me/owned-products/` существует в schema, но фактически `405`.
- `products` CRUD сейчас открыт всем authenticated (если на фронте не нужен, скрыть в UI).
- CORS настройки в `settings.py` не заданы (нужна same-origin схема или отдельная настройка).

