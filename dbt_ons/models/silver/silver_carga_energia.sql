-- Camada Silver: limpeza, padronização de tipos/datas e deduplicação
-- Conforme Dicionário de Dados do ONS, val_cargaenergiamwmed não permite nulos, zeros ou negativos.

WITH raw_carga AS (
    SELECT
        TRIM(CAST(id_subsistema AS VARCHAR)) AS id_subsistema,
        UPPER(TRIM(CAST(nom_subsistema AS VARCHAR))) AS nom_subsistema,
        CAST(din_instante AS DATE) AS data_referencia,
        CAST(din_instante AS TIMESTAMP) AS instante_original,
        TRY_CAST(val_cargaenergiamwmed AS DOUBLE) AS carga_mwmed
    FROM read_parquet('../data/bronze/CARGA_ENERGIA_*.parquet')
),

valid_carga AS (
    SELECT
        id_subsistema,
        nom_subsistema,
        data_referencia,
        carga_mwmed
    FROM raw_carga
    WHERE id_subsistema IS NOT NULL
      AND data_referencia IS NOT NULL
      AND carga_mwmed IS NOT NULL
      AND carga_mwmed > 0
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY id_subsistema, data_referencia
        -- Desempate determinístico: mantém o registro com o instante mais recente e,
        -- persistindo o empate, o de maior carga.
        ORDER BY instante_original DESC, carga_mwmed DESC
    ) = 1
)

SELECT
    id_subsistema,
    nom_subsistema,
    data_referencia,
    carga_mwmed
FROM valid_carga
ORDER BY data_referencia ASC, id_subsistema ASC
