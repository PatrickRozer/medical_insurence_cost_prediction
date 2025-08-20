CREATE TABLE medical_insurance (
    age INT,
    sex VARCHAR(10),
    bmi FLOAT,
    children INT,
    smoker VARCHAR(10),
    region VARCHAR(20),
    charges FLOAT
);

ALTER TABLE public."Medical_Insurance"
  ADD COLUMN IF NOT EXISTS predicted_cost numeric(12,2),
  ADD COLUMN IF NOT EXISTS created_at timestamp without time zone DEFAULT now();
ALTER TABLE "Medical_Insurance"
ALTER COLUMN charges TYPE FLOAT
USING charges::FLOAT;
ALTER TABLE "Medical_Insurance"
ALTER COLUMN charges TYPE FLOAT
USING bmi::FLOAT;
-- Average insurance charges
SELECT AVG(charges) AS avg_charges FROM "Medical_Insurance";

-- Smokers vs non-smokers
SELECT smoker, COUNT(*) FROM "Medical_Insurance" GROUP BY smoker;

-- Average BMI
SELECT AVG(bmi::FLOAT) AS avg_bmi
FROM "Medical_Insurance";


-- Region-wise count
SELECT region, COUNT(*) FROM "Medical_Insurance" GROUP BY region;

-- Avg charges by smoker
SELECT smoker, AVG(charges) AS avg_charges
FROM "Medical_Insurance" GROUP BY smoker;

-- Avg charges by gender
SELECT sex, AVG(charges) AS avg_charges
FROM "Medical_Insurance" GROUP BY sex;

-- Children vs charges
SELECT children, AVG(charges) AS avg_charges
FROM "Medical_Insurance" GROUP BY children ORDER BY children;

-- Smoking & age impact
SELECT age, smoker, AVG(charges) AS avg_charges
FROM "Medical_Insurance" GROUP BY age, smoker ORDER BY age;

-- Obese smokers (BMI > 30)
SELECT COUNT(*) AS obese_smokers,
       AVG(charges) AS avg_charges
FROM "Medical_Insurance"
WHERE smoker='yes' AND bmi::FLOAT > 30;
