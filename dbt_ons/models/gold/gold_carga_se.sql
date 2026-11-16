-- Camada Gold: Série diária limpa e contínua do subsistema Sudeste/Centro-Oeste (SE)

SELECT
    data_referencia,
    id_subsistema,
    nom_subsistema,
    carga_mwmed,
    EXTRACT(YEAR FROM data_referencia) AS ano,
    EXTRACT(MONTH FROM data_referencia) AS mes,
    EXTRACT(ISODOW FROM data_referencia) AS dia_semana_iso
FROM {{ ref('silver_carga_energia') }}
WHERE id_subsistema = 'SE'
ORDER BY data_referencia ASC
