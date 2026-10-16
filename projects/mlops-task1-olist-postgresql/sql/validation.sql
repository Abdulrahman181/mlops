SELECT
    table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

\set ON_ERROR_STOP on

DO $$
DECLARE
    missing_tables text;
BEGIN
    SELECT string_agg(required_table, ', ' ORDER BY required_table)
    INTO missing_tables
    FROM (
        SELECT unnest(ARRAY[
            'customers', 'geolocation', 'order_items', 'order_payments',
            'order_reviews', 'orders', 'product_category_translation',
            'products', 'sellers'
        ]) AS required_table
    ) AS required
    WHERE NOT EXISTS (
        SELECT 1
        FROM information_schema.tables AS available
        WHERE available.table_schema = 'public'
          AND available.table_name = required.required_table
    );

    IF missing_tables IS NOT NULL THEN
        RAISE EXCEPTION 'Validation stopped: required public tables are missing: %', missing_tables;
    END IF;
END $$;


SELECT
    'customers' AS table_name,
    COUNT(*) AS row_count
FROM customers

UNION ALL

SELECT
    'geolocation',
    COUNT(*)
FROM geolocation

UNION ALL

SELECT
    'order_items',
    COUNT(*)
FROM order_items

UNION ALL

SELECT
    'order_payments',
    COUNT(*)
FROM order_payments

UNION ALL

SELECT
    'order_reviews',
    COUNT(*)
FROM order_reviews

UNION ALL

SELECT
    'orders',
    COUNT(*)
FROM orders

UNION ALL

SELECT
    'products',
    COUNT(*)
FROM products

UNION ALL

SELECT
    'sellers',
    COUNT(*)
FROM sellers

UNION ALL

SELECT
    'product_category_translation',
    COUNT(*)
FROM product_category_translation

ORDER BY table_name;


SELECT
    o.order_id,
    o.order_status,
    o.order_purchase_timestamp,
    c.customer_city,
    c.customer_state
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
LIMIT 10;


SELECT
    o.order_id,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value
FROM orders AS o
JOIN order_items AS oi
    ON o.order_id = oi.order_id
LIMIT 10;


SELECT COUNT(*) AS orders_without_customer
FROM orders AS o
LEFT JOIN customers AS c
    ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;


SELECT COUNT(*) AS items_without_order
FROM order_items AS oi
LEFT JOIN orders AS o
    ON oi.order_id = o.order_id
WHERE o.order_id IS NULL;


SELECT COUNT(*) AS items_without_product
FROM order_items AS oi
LEFT JOIN products AS p
    ON oi.product_id = p.product_id
WHERE p.product_id IS NULL;


SELECT COUNT(*) AS items_without_seller
FROM order_items AS oi
LEFT JOIN sellers AS s
    ON oi.seller_id = s.seller_id
WHERE s.seller_id IS NULL;


SELECT COUNT(*) AS payments_without_order
FROM order_payments AS op
LEFT JOIN orders AS o
    ON op.order_id = o.order_id
WHERE o.order_id IS NULL;


SELECT
    COUNT(*) AS total_reviews,
    COUNT(DISTINCT review_id) AS unique_review_ids,
    COUNT(*) - COUNT(DISTINCT review_id) AS duplicate_review_ids
FROM order_reviews;


SELECT
    COUNT(*) AS delivered_orders
FROM orders
WHERE order_status = 'delivered';


SELECT
    COUNT(*) AS delivered_orders_with_delivery_date
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL;


SELECT
    COUNT(*) AS late_delivered_orders
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL
  AND order_delivered_customer_date > order_estimated_delivery_date;


SELECT
    COUNT(*) AS on_time_delivered_orders
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL
  AND order_delivered_customer_date <= order_estimated_delivery_date;
