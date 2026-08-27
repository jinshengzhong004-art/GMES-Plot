"""Versioned cartographic standards and stratigraphic vocabulary metadata."""

STYLE_LIBRARY_VERSION = "2026.08"
VERIFIED_ON = "2026-08-17"

STANDARD_PROFILES = (
    {
        "id": "DZT-0069-2024",
        "title": "DZ/T 0069-2024 地球物理勘查图图式图例及色标",
        "status": "现行",
        "scope": "物探图件的图式、图例、用色和整饰主依据",
        "verified_on": VERIFIED_ON,
        "url": "https://std.samr.gov.cn/hb/search/stdHBDetailedCNF?id=32F4462735DF5C62E06397BE0A0A5112",
    },
    {
        "id": "GBT-958-2015",
        "title": "GB/T 958-2015 区域地质图图例",
        "status": "现行；2025-11-03复审结论为继续有效",
        "scope": "区域地质单位、构造和基础地质图例",
        "verified_on": VERIFIED_ON,
        "url": "https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=2C6718A842C6D8CE39587B60410C3BD8",
    },
    {
        "id": "GBT-12328-1990",
        "title": "GB/T 12328-1990 综合工程地质图图例及色标",
        "status": "现行但正在修订，不作为永久固定模板",
        "scope": "工程地质图例兼容参考",
        "verified_on": VERIFIED_ON,
        "url": "https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=D9A4CD3D6599D8381B1337C4E61E9720",
    },
    {
        "id": "GBT-17412.1-3-1998",
        "title": "GB/T 17412.1～3-1998 岩石分类和命名方案",
        "status": "三部分现行但均有修订计划，名称库必须版本化",
        "scope": "岩浆岩、沉积岩、变质岩名称分类参考",
        "verified_on": VERIFIED_ON,
        "url": "https://std.samr.gov.cn/gb/search/gbDetailed?id=71F772D79482D3A7E05397BE0A0AB82A",
    },
)

STRATIGRAPHY_PROFILE = {
    "id": "ICS-2026-06",
    "title": "ICS International Chronostratigraphic Chart 2026/06",
    "license": "CC BY 4.0",
    "verified_on": VERIFIED_ON,
    "url": "https://stratigraphy.org/ICSchart/ChronostratChart2026-06.pdf",
}

# Compact project tree. Colours follow the familiar ICS/CCGM system-level palette;
# exact publication output remains governed by the selected standards profile.
STRATIGRAPHY_TREE = (
    ("显生宙 Phanerozoic", "#ffffff", (
        ("新生代 Cenozoic", "#f2f91d", (
            ("第四纪 Quaternary", "#fff2ae", ()),
            ("新近纪 Neogene", "#ffe619", ()),
            ("古近纪 Paleogene", "#fd9a52", ()),
        )),
        ("中生代 Mesozoic", "#67c5ca", (
            ("白垩纪 Cretaceous", "#7fc64e", ()),
            ("侏罗纪 Jurassic", "#34b2c9", ()),
            ("三叠纪 Triassic", "#812b92", ()),
        )),
        ("古生代 Paleozoic", "#99c08d", (
            ("二叠纪 Permian", "#f04028", ()),
            ("石炭纪 Carboniferous", "#67a599", ()),
            ("泥盆纪 Devonian", "#cb8c37", ()),
            ("志留纪 Silurian", "#b3e1b6", ()),
            ("奥陶纪 Ordovician", "#009270", ()),
            ("寒武纪 Cambrian", "#7fa056", ()),
        )),
    )),
    ("元古宙 Proterozoic", "#f7b5d3", (
        ("新元古代 Neoproterozoic", "#f4a7c5", ()),
        ("中元古代 Mesoproterozoic", "#f39bc1", ()),
        ("古元古代 Paleoproterozoic", "#ee8ab5", ()),
    )),
    ("太古宙 Archean", "#f0047f", ()),
    ("冥古宙 Hadean", "#ae027e", ()),
)

