import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
const experiment = JSON.parse(
  fs.readFileSync(path.resolve("public/research/experiment.json"), "utf8"),
);
const qaDir = process.env.QA_DIR || path.join(os.tmpdir(), "smartenergy-qa");
fs.mkdirSync(qaDir, { recursive: true });
const format = (v: number) =>
  v.toLocaleString("en-US", {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  });

test("production subpath, genuine model/origin changes, dates, range and CSV download", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning") errors.push(m.text());
  });
  page.on("response", (r) => {
    if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`);
  });
  await page.goto("./");
  await expect(page).toHaveTitle("SmartEnergy — Energy Forecasting Lab");
  await expect(
    page.getByRole("heading", { name: "Explore the next hour." }),
  ).toBeVisible();
  await expect(page.locator(".recharts-line-curve").first()).toBeVisible();
  await expect(page.getByTestId("prediction")).toContainText(
    format(experiment.predictions[0][experiment.metadata.selected_ml]),
  );
  await page.screenshot({
    path: path.join(qaDir, "desktop.png"),
    fullPage: true,
  });
  await page.getByLabel("Model", { exact: true }).selectOption("Persistence");
  await expect(page.getByTestId("prediction")).toContainText(
    format(experiment.predictions[0].Persistence),
  );
  await page
    .getByRole("button", { name: "Next forecast", exact: true })
    .click();
  await expect(page.getByTestId("prediction")).toContainText(
    format(experiment.predictions[1].Persistence),
  );
  await page.getByLabel("Forecast origin", { exact: true }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByTestId("prediction")).toContainText(
    format(experiment.predictions[2].Persistence),
  );
  await page
    .getByLabel("Forecast date")
    .selectOption(experiment.predictions[200].origin.slice(0, 10));
  await expect(page.getByLabel("Forecast origin", { exact: true })).toHaveValue(
    "0",
  );
  await page.getByRole("button", { name: "48h", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "48h", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  const sharedPrediction = await page.getByTestId("prediction").textContent();
  await page.reload();
  await expect(page.getByTestId("prediction")).toHaveText(sharedPrediction!);
  await expect(page.getByLabel("Model", { exact: true })).toHaveValue(
    "Persistence",
  );
  await expect(
    page.getByRole("button", { name: "48h", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("link", { name: "Reproduce", exact: true }).click();
  const downloadPromise = page.waitForEvent("download");
  await page
    .getByRole("link", { name: "All held-out predictions (CSV)" })
    .click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("test_predictions.csv");
  const downloaded = await download.path();
  expect(fs.readFileSync(downloaded!, "utf8").split("\n")).toHaveLength(
    experiment.predictions.length + 2,
  );
  expect(errors).toEqual([]);
});

test("model comparison, validation, metrics, ablation and method controls", async ({
  page,
}) => {
  await page.goto("./");
  await page
    .getByRole("link", { name: "Models & insights", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Earn the extra complexity." }),
  ).toBeVisible();
  await page.getByLabel("Evaluation period").selectOption("validation");
  const selected = experiment.metrics.validation.find(
    (r: { model: string }) => r.model === experiment.metadata.selected_model,
  );
  await expect(page.locator("tr.selected-row").first()).toContainText(
    format(selected.mae),
  );
  await page.getByLabel("Comparison metric").selectOption("rmse");
  await expect(
    page.getByRole("img", { name: "validation comparison by rmse" }),
  ).toBeVisible();
  await page.getByLabel("Comparison metric").selectOption("r2");
  await expect(
    page.getByRole("img", { name: "validation comparison by r2" }),
  ).toBeVisible();
  await expect(
    page.getByText("History + time + sensors", { exact: true }),
  ).toBeVisible();
  await page
    .locator("summary")
    .filter({ hasText: "Gradient boosting" })
    .click();
  await expect(page.getByText("early_stopping", { exact: false })).toHaveCount(
    0,
  );
  await page.getByRole("link", { name: "Methodology", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "A forecast you can audit." }),
  ).toBeVisible();
  await page.locator("summary").filter({ hasText: "Inspect all" }).click();
  await expect(
    page.getByRole("rowheader", { name: "energy_lag_144", exact: true }),
  ).toBeVisible();
});

test("data views render and all research figures resolve", async ({ page }) => {
  await page.goto("./");
  await page.getByRole("link", { name: "Data", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "A household, measured over time." }),
  ).toBeVisible();
  for (const view of ["hourly", "distribution", "daily"]) {
    await page.getByLabel("Data view").selectOption(view);
    await expect(page.locator(".recharts-surface").first()).toBeVisible();
  }
  await page.getByLabel("Data period").selectOption("2016-03");
  await expect(page.getByLabel("Data period")).toHaveValue("2016-03");
  const images = page.locator("figure img");
  await expect(images).toHaveCount(6);
  for (const img of await images.all()) {
    const r = await page.request.get((await img.getAttribute("src"))!);
    expect(r.ok()).toBeTruthy();
  }
});

test("error filters change real samples and a failure opens the correct forecast", async ({
  page,
}) => {
  await page.goto("./");
  await page
    .getByRole("link", { name: "Error laboratory", exact: true })
    .click();
  await expect(page.getByTestId("filter-summary")).toContainText(
    experiment.predictions.length.toLocaleString("en-US"),
  );
  await page.getByLabel("Time of day").selectOption("night");
  await page.getByLabel("Consumption", { exact: true }).selectOption("high");
  const filtered = experiment.predictions.filter(
    (p: { origin: string; actual: number }) => {
      const h = Number(p.origin.slice(11, 13));
      return (
        (h < 7 || h >= 22) &&
        p.actual >= experiment.metadata.thresholds_train_only.high
      );
    },
  );
  await expect(page.getByTestId("filter-summary")).toContainText(
    `${filtered.length.toLocaleString("en-US")} origins`,
  );
  await page.getByLabel("Model", { exact: true }).selectOption("Random forest");
  await expect(page.getByTestId("filter-summary")).toContainText(
    "Random forest",
  );
  await page
    .getByRole("button", { name: "Open forecast", exact: true })
    .first()
    .click();
  await expect(
    page.getByRole("heading", { name: "Explore the next hour." }),
  ).toBeVisible();
  await expect(page.getByLabel("Model", { exact: true })).toHaveValue(
    "Random forest",
  );
});

test("desktop accessibility and mobile sections have no overflow or runtime failures", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  await expect(page.getByTestId("prediction")).toBeVisible();
  const desktopA11y = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(desktopA11y.violations).toEqual([]);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: path.join(qaDir, "mobile.png"),
    fullPage: true,
  });
  for (const name of [
    "Forecast explorer",
    "Data",
    "Models & insights",
    "Error laboratory",
    "Methodology",
    "Reproduce",
  ]) {
    await page.getByRole("link", { name, exact: true }).click();
    await expect(page.locator("h1")).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBeTruthy();
    const a11y = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
      .analyze();
    expect(a11y.violations).toEqual([]);
  }
  expect(errors).toEqual([]);
});
