# EDA Findings

## 1. Dataset Overview

The training data contains three independent sources and one ground-truth file.

| Dataset | Rows | Columns |
|---|---:|---:|
| Source 1 | 2,206,821 | 4 |
| Source 2 | 5,034,616 | 4 |
| Source 3 | 5,285,603 | 4 |
| Ground Truth | 2,206,821 | 2 |

All three source files contain:
- entity_id
- business_name
- business_address
- country

The ground-truth file contains:
- source1_entity_id
- matched_entity_ids

---

## 2. Country Distribution

### Source 1

| Country | Records |
|---|---:|
| US | 1,323,633 |
| India | 883,188 |

### Source 2

| Country | Records |
|---|---:|
| US | 3,016,817 |
| India | 2,017,799 |

### Source 3

| Country | Records |
|---|---:|
| US | 3,170,056 |
| India | 2,115,547 |

The training data contains businesses from the US and India.

---

## 3. Singleton / No-Match Analysis

There are **123,247 Source 1 entities** with no matching Source 2 or Source 3 record.

Total Source 1 entities: **2,206,821**

Therefore, approximately **5.58%** of Source 1 entities have no match.

This is important because the matching system must be able to correctly identify entities with no valid match instead of forcing an incorrect match.

---

## 4. Match Count Distribution

| Number of Matches | Source 1 Entities |
|---|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3+ | 1,589,205 |

A large proportion of Source 1 entities have multiple matching records. Therefore, the solution must support one-to-many matching rather than assuming that every Source 1 entity has at most one match.

---

## 5. Target ID Reuse Check

The ground-truth data was checked to determine whether the same Source 2 or Source 3 entity ID is assigned to multiple Source 1 entities.

Result:

**0 Source 2/Source 3 IDs were found to be matched to multiple Source 1 entities.**

This indicates that the ground truth does not reuse the same target entity across multiple Source 1 entities.

---

## 6. Real Matching Examples and Observed Noise

Several real matched pairs were inspected to understand the type of noise present in the data.

Observed patterns include:

### Typos

Examples include:
- `Williams` → `Wilblims`
- `Wayne` → `Wanye`
- `Township` → `Townshiip`

### Name truncation or variation

Example:
- `Maure Williams Colombier Inc`
- `Maure Williams Colombier`
- `Maure Williams Inc Center`

These records can refer to the same business even though the names are not identical.

### Website-style representation

Example:
- `Maure Williams Colombier`
- `maurewilliamscolombier.com`

### Multilingual / Script Variation

Example:
- `Raj Investments LLP`
- Tamil-script representations of `Raj Investments LLP`

This shows that matching cannot depend only on exact English string equality.

### Missing Values

Some matched records have missing business addresses.

### Address Noise

Examples include:
- different word ordering
- abbreviations
- capitalization differences
- spelling errors
- noisy components such as `null`
- variations such as `45th` → `45ND`

These observations support the need for robust name and address normalization and fuzzy similarity features.

---

## 7. Postcode Coverage

A simple postcode extraction analysis was performed on the business addresses.

| Source | Records with Postcode | Coverage |
|---|---:|---:|
| Source 1 | 147,257 | 6.67% |
| Source 2 | 369,140 | 7.33% |
| Source 3 | 385,727 | 7.30% |

Postcodes are available for only a small percentage of records.

Therefore, postcode should be treated as a **supporting matching feature** rather than the primary blocking or matching signal.

---

## 8. Key EDA Findings

The EDA indicates that:

1. Source 1 contains 2.2M business entities, while Sources 2 and 3 contain substantially more records.
2. Approximately 5.58% of Source 1 entities have no match.
3. Many Source 1 entities have multiple matching records, so one-to-many matching must be supported.
4. Business names contain typos, abbreviations, truncations, punctuation differences, multilingual representations and website-style names.
5. Business addresses contain missing values, spelling errors, abbreviations, reordering and noisy components.
6. Postcodes have low coverage of approximately 7%, so they should not be relied upon as the main matching signal.
7. The matching pipeline should therefore combine normalization, blocking and multiple similarity features rather than relying on exact matching.