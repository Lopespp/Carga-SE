-- Teste singular do dbt: garante que nenhuma carga diária na camada Gold seja <= 0
SELECT
    data_referencia,
    carga_mwmed
FROM {{ ref('gold_carga_se') }}
WHERE carga_mwmed <= 0
