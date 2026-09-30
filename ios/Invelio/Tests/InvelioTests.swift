import XCTest
@testable import Invelio

final class InvelioTests: XCTestCase {
    func testStockDecoding() throws {
        let json = """
        {"ticker": "BBCA", "name": "Bank Central Asia", "price": 9875.0, "change": 1.25}
        """.data(using: .utf8)!

        let stock = try JSONDecoder().decode(Stock.self, from: json)
        XCTAssertEqual(stock.ticker, "BBCA")
        XCTAssertEqual(stock.price, 9875.0)
    }

    func testHoldingLotCalculations() throws {
        let lot = HoldingLot(
            ticker: "BBCA.JK",
            stockName: "Bank Central Asia",
            market: "IDX",
            currency: "IDR",
            pricePerShare: 10000,
            totalInvested: 5000000
        )

        XCTAssertEqual(lot.ticker, "BBCA.JK")
        XCTAssertEqual(lot.shares, 500.0)
        XCTAssertEqual(lot.lots, 5.0)
        XCTAssertEqual(lot.symbol, "BBCA")
    }

    func testLotBasedPortfolioCalculation() throws {
        // Example from user: price 1 share = 6000, 1 lot => 100 shares, portfolio = 600,000
        let lot = HoldingLot(
            ticker: "BBCA.JK",
            stockName: "Bank Central Asia",
            market: "IDX",
            currency: "IDR",
            pricePerShare: 6000,
            lots: 1.0
        )

        XCTAssertEqual(lot.lots, 1.0)
        XCTAssertEqual(lot.shares, 100.0)
        XCTAssertEqual(lot.totalInvested, 600000.0)
    }

    func testPurchaseFormEntryLotInput() throws {
        let entry = PurchaseFormEntry(
            priceInput: "6000",
            lotInput: "1"
        )

        XCTAssertEqual(entry.price, 6000.0)
        XCTAssertEqual(entry.lots, 1.0)
        XCTAssertEqual(entry.shares, 100.0)
        XCTAssertEqual(entry.total, 600000.0)
        XCTAssertEqual(entry.formattedShares, "100")
        XCTAssertEqual(entry.formattedLots, "1")
    }
}
